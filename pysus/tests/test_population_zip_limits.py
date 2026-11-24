"""Limits for population zip members fetched over anonymous FTP.

``get_population`` downloads a zip from the DataSUS FTP server and
``_unzip_to_dataframe`` extracts the first spreadsheet or database member.
These tests are the spec for that path. They fail while a member can be
extracted and loaded with no cap.

A member must be refused when either of these is true:

* uncompressed size is greater than ``MAX_UNCOMPRESSED_BYTES`` (2 MiB)
* ``file_size / max(compress_size, 1)`` is greater than
  ``MAX_COMPRESSION_RATIO`` (50)

Refusal means the helper raises, reads at most the uncompressed cap from
the member (and reads nothing when only the ratio is exceeded), and does
not create the extracted member or any other file outside the download
cache. A table of a few dozen rows under both limits must still load.

The suite talks to a local FTP peer so the real client code runs without
contacting ftp.datasus.gov.br. The peer is the stand-in for an altered
cleartext response; ``ftplib.FTP`` itself is not stubbed.
"""

from __future__ import annotations

import datetime
import os
import socket
import struct
import tempfile
import threading
import traceback
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipExtFile, ZipFile

# Bound the download cache before pysus.ftp reads the environment.
CACHE_DIR = tempfile.mkdtemp(prefix="pysus-population-cache-")
os.environ["PYSUS_CACHEPATH"] = CACHE_DIR

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from pysus.ftp import File  # noqa: E402
from pysus.ftp.utils import zfill_year  # noqa: E402

MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024
MAX_COMPRESSION_RATIO = 50

POP_DIR = "/dissemin/publicos/IBGE/POP"
POPTCU_DIR = "/dissemin/publicos/IBGE/POPTCU"
CENSO_DIR = "/dissemin/publicos/IBGE/censo"

_SERVER = None
_RESTORE_CONNECT = None
IBGE = None


def _pick_years(count: int) -> list[int]:
    """Years whose two-digit form round-trips through ``zfill_year``."""
    year = datetime.datetime.now().year
    chosen = []
    while len(chosen) < count and year > 1990:
        if zfill_year(year) == year:
            chosen.append(year)
        year -= 1
    if len(chosen) < count:
        raise RuntimeError("could not pick stable population years")
    return chosen


def _zip_bytes(members: list[tuple[str, bytes]], compression: int) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=compression) as archive:
        for name, payload in members:
            archive.writestr(name, payload)
    return buffer.getvalue()


def _make_dbf(num_records: int, field_len: int = 16) -> bytes:
    """Minimal dBASE III table of one character field."""
    name = b"POP".ljust(11, b"\x00")
    field = struct.pack(
        "<11scLBBHBBBB7sB",
        name,
        b"C",
        0,
        field_len,
        0,
        0,
        0,
        0,
        0,
        0,
        b"\x00" * 7,
        0,
    )
    header_len = 32 + len(field) + 1
    record_len = 1 + field_len
    header = struct.pack(
        "<BBBBLHHHBBLLLBBH",
        3,
        24,
        1,
        15,
        num_records,
        header_len,
        record_len,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        0,
    )
    record = b" " + (b"1" * field_len)
    return header + field + b"\r" + (record * num_records) + b"\x1a"


def _repeated_csv(uncompressed: int) -> bytes:
    line = b"0\n"
    copies = (uncompressed + len(line) - 1) // len(line)
    return line * copies


def _build_archives(years: list[int]) -> dict[str, bytes]:
    """Map absolute FTP paths to the zip each population request downloads."""
    (
        safe_year,
        size_year,
        ratio_year,
        dbf_year,
        dbc_year,
        nested_year,
        censo_year,
    ) = years

    safe_rows = ["municipio,habitantes\n"]
    for index in range(40):
        # Distinct values so the deflate ratio stays under the cap.
        safe_rows.append(f"{index},{100000 + index * 97}\n")
    safe_csv = "".join(safe_rows).encode()

    size_csv = _repeated_csv(MAX_UNCOMPRESSED_BYTES + 64 * 1024)
    ratio_csv = _repeated_csv(256 * 1024)
    dbf_blob = _make_dbf(num_records=4000, field_len=16)
    dbc_blob = b"\x00" * (256 * 1024)

    specs = {
        safe_year: _zip_bytes([("populacao.csv", safe_csv)], ZIP_DEFLATED),
        size_year: _zip_bytes([("populacao.csv", size_csv)], ZIP_STORED),
        ratio_year: _zip_bytes(
            [("populacao.csv", ratio_csv)], ZIP_DEFLATED
        ),
        dbf_year: _zip_bytes([("populacao.DBF", dbf_blob)], ZIP_DEFLATED),
        dbc_year: _zip_bytes([("populacao.dbc", dbc_blob)], ZIP_DEFLATED),
        nested_year: _zip_bytes(
            [
                ("leia-me.txt", b"not a table"),
                ("dados/populacao.CSV", ratio_csv),
            ],
            ZIP_DEFLATED,
        ),
        censo_year: _zip_bytes(
            [("CENSOALF.csv", ratio_csv)], ZIP_DEFLATED
        ),
    }

    files: dict[str, bytes] = {}
    for year, blob in specs.items():
        suffix = f"{year % 100:02d}"
        name = f"POPTBR{suffix}.zip"
        if year == censo_year:
            censo_name = f"CENSOALF{suffix}.zip"
            files[f"{CENSO_DIR}/{censo_name}"] = blob
            continue
        for directory in (POP_DIR, POPTCU_DIR):
            files[f"{directory}/{name}"] = blob
    return files


class _FtpPeer:
    """Minimal anonymous FTP peer. Speaks the commands ftplib sends."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.commands: list[str] = []
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(16)
        sock.settimeout(0.2)
        self._sock = sock
        self.port = sock.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=2)

    def _remember(self, line: str) -> None:
        with self._lock:
            self.commands.append(line)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._session, args=(conn,), daemon=True
            ).start()

    def _session(self, conn: socket.socket) -> None:
        pasv = None
        try:
            conn.sendall(b"220 population-test\r\n")
            reader = conn.makefile("r", encoding="utf-8", newline="\r\n")
            cwd = "/"
            while not self._stop.is_set():
                line = reader.readline()
                if not line:
                    break
                text = line.strip()
                self._remember(text)
                command, _, arg = text.partition(" ")
                command = command.upper()
                if command == "USER":
                    conn.sendall(b"331 password required\r\n")
                elif command == "PASS":
                    conn.sendall(b"230 logged in\r\n")
                elif command in {"TYPE", "NOOP", "MODE", "STRU", "OPTS"}:
                    conn.sendall(b"200 ok\r\n")
                elif command == "SYST":
                    conn.sendall(b"215 UNIX\r\n")
                elif command == "FEAT":
                    conn.sendall(b"211 end\r\n")
                elif command == "PWD":
                    conn.sendall(b'257 "/"\r\n')
                elif command == "CWD":
                    cwd = arg or "/"
                    conn.sendall(b"250 cwd\r\n")
                elif command == "PASV":
                    if pasv is not None:
                        pasv.close()
                    pasv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    pasv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    pasv.bind(("127.0.0.1", 0))
                    pasv.listen(1)
                    pasv.settimeout(10)
                    port = pasv.getsockname()[1]
                    host = f"127,0,0,1,{port // 256},{port % 256}"
                    conn.sendall(
                        f"227 Entering Passive Mode ({host})\r\n".encode()
                    )
                elif command in {"LIST", "RETR"}:
                    pasv = self._transfer(conn, pasv, command, arg, cwd)
                elif command == "QUIT":
                    conn.sendall(b"221 bye\r\n")
                    break
                else:
                    conn.sendall(b"502 unsupported\r\n")
        except Exception:
            self.errors.append(traceback.format_exc())
        finally:
            if pasv is not None:
                pasv.close()
            conn.close()

    def _transfer(self, conn, pasv, command: str, arg: str, cwd: str):
        if pasv is None:
            conn.sendall(b"425 use pasv\r\n")
            return None
        if command == "RETR" and arg not in self.files:
            conn.sendall(b"550 missing\r\n")
            pasv.close()
            return None
        if command == "LIST":
            payload = self._list(arg, cwd)
        else:
            payload = self.files[arg]
        data, _addr = pasv.accept()
        try:
            conn.sendall(b"150 opening data connection\r\n")
            data.sendall(payload)
        finally:
            data.close()
            pasv.close()
        conn.sendall(b"226 transfer complete\r\n")
        return None

    def _list(self, path: str, cwd: str) -> bytes:
        folder = (path or cwd).rstrip("/") or "/"
        lines = []
        for full, blob in self.files.items():
            parent, name = full.rsplit("/", 1)
            if parent != folder:
                continue
            size = len(blob)
            lines.append(f"01-15-26  10:00AM {size:>16} {name}\r\n")
        return "".join(lines).encode()


def _redirect_datasus(port: int):
    original = socket.create_connection

    def connect(address, *args, **kwargs):
        host = address[0]
        if host == "ftp.datasus.gov.br":
            address = ("127.0.0.1", port)
        return original(address, *args, **kwargs)

    socket.create_connection = connect
    return original


def _start_peer():
    global _SERVER, _RESTORE_CONNECT, IBGE, YEARS, FILES
    years = _pick_years(7)
    files = _build_archives(years)
    _SERVER = _FtpPeer(files)
    _RESTORE_CONNECT = _redirect_datasus(_SERVER.port)
    from pysus.online_data import IBGE as imported

    IBGE = imported
    YEARS = years
    FILES = files


def _stop_peer():
    global _RESTORE_CONNECT, _SERVER
    if _RESTORE_CONNECT is not None:
        socket.create_connection = _RESTORE_CONNECT
        _RESTORE_CONNECT = None
    if _SERVER is not None:
        _SERVER.close()
        _SERVER = None


class _Call:
    def __init__(self, result, created: list[str], member_bytes: int):
        self.result = result
        self.created = created
        self.member_bytes = member_bytes


def _outside_cache(path: str) -> bool:
    absolute = os.path.abspath(path)
    cache = os.path.abspath(CACHE_DIR)
    if absolute == cache or absolute.startswith(cache + os.sep):
        return False
    temporary = os.path.abspath(tempfile.gettempdir())
    return absolute.startswith(temporary + os.sep)


def _call_population(year: int, source: str = "POP", **kwargs) -> _Call:
    """Run one request while recording member reads and new temp files."""
    created: list[str] = []
    read_bytes = {"n": 0}
    real_open = open
    real_read = ZipExtFile.read

    def tracking_open(*args, **kwargs):
        stream = real_open(*args, **kwargs)
        mode = "r"
        if len(args) >= 2 and isinstance(args[1], str):
            mode = args[1]
        if "mode" in kwargs:
            mode = kwargs["mode"]
        writing = any(flag in mode for flag in ("w", "a", "x", "+"))
        target = args[0] if args else None
        if writing and isinstance(target, (str, os.PathLike)):
            path = os.fspath(target)
            if _outside_cache(path):
                created.append(path)
        return stream

    def tracking_read(self, n=-1):
        data = real_read(self, n)
        if isinstance(data, (bytes, bytearray)):
            read_bytes["n"] += len(data)
        return data

    builtins_open = __import__("builtins").open
    import builtins

    real_download = File.download

    def download(self, local_dir=CACHE_DIR, _pbar=None):
        # ``File.download`` binds its cache default when pysus is imported,
        # which is before this module can set PYSUS_CACHEPATH. Pass the
        # cache explicitly and still run the real retrbinary path.
        return real_download(self, CACHE_DIR, _pbar)

    builtins.open = tracking_open
    ZipExtFile.read = tracking_read
    File.download = download
    try:
        try:
            result = IBGE.get_population(year=year, source=source, **kwargs)
        except Exception as exc:  # refusal is the expected bomb outcome
            result = exc
    finally:
        builtins.open = builtins_open
        ZipExtFile.read = real_read
        File.download = real_download
    return _Call(result, created, read_bytes["n"])


def _ratio(blob: bytes, name: str) -> tuple[int, float]:
    with ZipFile(BytesIO(blob)) as archive:
        info = archive.getinfo(name)
    return info.file_size, info.file_size / max(info.compress_size, 1)


class TestPopulationZipLimits:
    def test_small_table_still_loads(self):
        year = YEARS[0]
        blob = FILES[f"{POP_DIR}/POPTBR{year % 100:02d}.zip"]
        size, ratio = _ratio(blob, "populacao.csv")
        assert size <= MAX_UNCOMPRESSED_BYTES
        assert ratio <= MAX_COMPRESSION_RATIO

        outcome = _call_population(year, source="POP")
        assert _SERVER.errors == []
        assert any(
            command == "USER anonymous" for command in _SERVER.commands
        )
        assert isinstance(outcome.result, pd.DataFrame)
        assert len(outcome.result) == 40
        assert int(outcome.result.iloc[0]["habitantes"]) == 100000
        cached = Path(CACHE_DIR) / f"POPTBR{year % 100:02d}.zip"
        assert cached.read_bytes() == blob
        assert 0 < outcome.member_bytes <= MAX_UNCOMPRESSED_BYTES

    def test_oversized_stored_member_is_refused_before_extract(self):
        year = YEARS[1]
        blob = FILES[f"{POP_DIR}/POPTBR{year % 100:02d}.zip"]
        size, ratio = _ratio(blob, "populacao.csv")
        assert size > MAX_UNCOMPRESSED_BYTES
        assert ratio <= MAX_COMPRESSION_RATIO

        outcome = _call_population(year, source="POP")
        assert isinstance(outcome.result, Exception)
        assert outcome.created == []
        assert outcome.member_bytes <= MAX_UNCOMPRESSED_BYTES

    def test_high_ratio_csv_is_refused_before_read(self):
        year = YEARS[2]
        blob = FILES[f"{POP_DIR}/POPTBR{year % 100:02d}.zip"]
        size, ratio = _ratio(blob, "populacao.csv")
        assert size <= MAX_UNCOMPRESSED_BYTES
        assert ratio > MAX_COMPRESSION_RATIO

        outcome = _call_population(year, source="POP")
        assert isinstance(outcome.result, Exception)
        assert outcome.created == []
        assert outcome.member_bytes == 0

    def test_high_ratio_dbf_is_refused_before_read(self):
        year = YEARS[3]
        blob = FILES[f"{POP_DIR}/POPTBR{year % 100:02d}.zip"]
        size, ratio = _ratio(blob, "populacao.DBF")
        assert size <= MAX_UNCOMPRESSED_BYTES
        assert ratio > MAX_COMPRESSION_RATIO

        outcome = _call_population(year, source="POP")
        assert isinstance(outcome.result, Exception)
        assert outcome.created == []
        assert outcome.member_bytes == 0

    def test_high_ratio_dbc_is_refused_before_read(self):
        year = YEARS[4]
        blob = FILES[f"{POP_DIR}/POPTBR{year % 100:02d}.zip"]
        size, ratio = _ratio(blob, "populacao.dbc")
        assert size <= MAX_UNCOMPRESSED_BYTES
        assert ratio > MAX_COMPRESSION_RATIO

        outcome = _call_population(year, source="POP")
        assert isinstance(outcome.result, Exception)
        assert outcome.created == []
        assert outcome.member_bytes == 0

    def test_data_member_after_readme_is_still_limited(self):
        year = YEARS[5]
        blob = FILES[f"{POP_DIR}/POPTBR{year % 100:02d}.zip"]
        size, ratio = _ratio(blob, "dados/populacao.CSV")
        assert size <= MAX_UNCOMPRESSED_BYTES
        assert ratio > MAX_COMPRESSION_RATIO

        outcome = _call_population(year, source="POP")
        assert isinstance(outcome.result, Exception)
        assert outcome.created == []
        assert outcome.member_bytes == 0

    def test_censo_zip_uses_the_same_limit(self):
        year = YEARS[6]
        suffix = f"{year % 100:02d}"
        blob = FILES[f"{CENSO_DIR}/CENSOALF{suffix}.zip"]
        size, ratio = _ratio(blob, "CENSOALF.csv")
        assert size <= MAX_UNCOMPRESSED_BYTES
        assert ratio > MAX_COMPRESSION_RATIO

        outcome = _call_population(year, source="censo", censo_data="ALF")
        assert isinstance(outcome.result, Exception)
        assert outcome.created == []
        assert outcome.member_bytes == 0


@pytest.fixture(scope="module", autouse=True)
def _population_ftp_module():
    _start_peer()
    yield
    _stop_peer()

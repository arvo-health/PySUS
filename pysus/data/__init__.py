import os
import struct
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dbfread import DBF
from pyreaddbc import dbc2dbf


def dbc_to_dbf(dbc: str, _pbar=None) -> str:
    """
    Parses DBC files into DBFs
    """
    path = Path(dbc)

    if path.suffix.lower() != ".dbc":
        raise ValueError(f"Not a DBC file: {path}")

    dbf = path.with_suffix(".dbf")

    if _pbar:
        _pbar.reset(total=1)
        _pbar.set_description(f"{dbf.name}")

    _parquet = path.with_suffix(".parquet")
    if _parquet.exists():
        path.unlink(missing_ok=True)
        dbf.unlink(missing_ok=True)
        return str(_parquet)

    if dbf.exists():
        path.unlink(missing_ok=True)
        return str(dbf)

    dbc2dbf(str(path), str(dbf))
    path.unlink()

    if _pbar:
        _pbar.update(1)

    return str(dbf)


# PKWare DCL ("implode") tables, as in Mark Adler's blast.c bundled with
# pyreaddbc. Each byte is a run: count (high nibble + 1) of a bit length.
_BLAST_MAXBITS = 13
_BLAST_LITLEN = (
    11, 124, 8, 7, 28, 7, 188, 13, 76, 4, 10, 8, 12, 10, 12, 10, 8, 23, 8,
    9, 7, 6, 7, 8, 7, 6, 55, 8, 23, 24, 12, 11, 7, 9, 11, 12, 6, 7, 22, 5,
    7, 24, 6, 11, 9, 6, 7, 22, 7, 11, 38, 7, 9, 8, 25, 11, 8, 11, 9, 12,
    8, 12, 5, 38, 5, 38, 5, 11, 7, 5, 6, 21, 6, 10, 53, 8, 7, 24, 10, 27,
    44, 253, 253, 253, 252, 252, 252, 13, 12, 45, 12, 45, 12, 61, 12, 45,
    44, 173,
)  # fmt: skip
_BLAST_LENLEN = (2, 35, 36, 53, 38, 23)
_BLAST_DISTLEN = (2, 20, 53, 230, 247, 151, 248)
_BLAST_BASE = (3, 2, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 40, 72, 136, 264)
_BLAST_EXTRA = (0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8)


def _blast_lookup(rep: tuple) -> list:
    """
    Maps every _BLAST_MAXBITS-bit window of the stream to the
    (symbol, code length) it starts with, or None for an invalid code
    """
    lengths = []
    for byte in rep:
        lengths.extend([byte & 15] * ((byte >> 4) + 1))

    count = [0] * (_BLAST_MAXBITS + 1)
    for length in lengths:
        count[length] += 1
    offs = [0] * (_BLAST_MAXBITS + 2)
    for length in range(1, _BLAST_MAXBITS):
        offs[length + 1] = offs[length] + count[length]
    symbols = [0] * len(lengths)
    for symbol, length in enumerate(lengths):
        if length:
            symbols[offs[length]] = symbol
            offs[length] += 1

    table = []
    for window in range(1 << _BLAST_MAXBITS):
        code = first = index = 0
        entry = None
        for length in range(1, _BLAST_MAXBITS + 1):
            # Codes are stored bit-inverted, least significant bit first
            code |= ((window >> (length - 1)) & 1) ^ 1
            if code < first + count[length]:
                entry = (symbols[index + code - first], length)
                break
            index += count[length]
            first = (first + count[length]) << 1
            code <<= 1
        table.append(entry)
    return table


@lru_cache(maxsize=None)
def _blast_tables() -> tuple:
    return (
        _blast_lookup(_BLAST_LITLEN),
        _blast_lookup(_BLAST_LENLEN),
        _blast_lookup(_BLAST_DISTLEN),
    )


def blast(data: bytes, max_size: int) -> bytes:
    """
    Decompresses a PKWare DCL stream, raising ValueError as soon as the
    output would grow past `max_size` bytes
    """
    lit_table, len_table, dist_table = _blast_tables()
    total_bits = len(data) * 8
    padded = bytes(data) + b"\x00" * 4
    out = bytearray()
    pos = 0

    def take(need: int) -> int:
        nonlocal pos
        if pos + need > total_bits:
            raise ValueError("Truncated DBC stream")
        i = pos >> 3
        value = int.from_bytes(padded[i : i + 4], "little") >> (pos & 7)
        pos += need
        return value & ((1 << need) - 1)

    def decode(table: list) -> int:
        nonlocal pos
        i = pos >> 3
        window = int.from_bytes(padded[i : i + 3], "little") >> (pos & 7)
        entry = table[window & ((1 << _BLAST_MAXBITS) - 1)]
        if entry is None or pos + entry[1] > total_bits:
            raise ValueError("Invalid or truncated DBC stream")
        pos += entry[1]
        return entry[0]

    lit = take(8)
    if lit > 1:
        raise ValueError("Invalid DBC literal flag")
    dict_bits = take(8)
    if not 4 <= dict_bits <= 6:
        raise ValueError("Invalid DBC dictionary size")

    while True:
        if take(1):
            symbol = decode(len_table)
            length = _BLAST_BASE[symbol] + take(_BLAST_EXTRA[symbol])
            if length == 519:
                break
            shift = 2 if length == 2 else dict_bits
            dist = (decode(dist_table) << shift) + take(shift) + 1
            if dist > len(out):
                raise ValueError("Invalid DBC distance")
            if len(out) + length > max_size:
                raise ValueError(
                    f"Decompressed DBC data exceeds {max_size} bytes"
                )
            start = len(out) - dist
            while length:
                chunk = out[start : start + min(length, dist)]
                out += chunk
                start += len(chunk)
                length -= len(chunk)
        else:
            symbol = decode(lit_table) if lit else take(8)
            if len(out) >= max_size:
                raise ValueError(
                    f"Decompressed DBC data exceeds {max_size} bytes"
                )
            out.append(symbol)

    return bytes(out)


def dbc_bytes_to_dbf(dbc: bytes, max_size: int) -> bytes:
    """
    Converts DBC contents into DBF contents in memory, refusing a DBF larger
    than `max_size` bytes before it is fully produced
    """
    if len(dbc) < 10:
        raise ValueError("Invalid DBC file: too short")
    header_len = int.from_bytes(dbc[8:10], "little")
    if header_len < 32 or len(dbc) < header_len + 4:
        raise ValueError(f"Invalid DBC header size {header_len}")
    if header_len > max_size:
        raise ValueError(f"DBF header exceeds {max_size} bytes")

    header = bytearray(dbc[:header_len])
    header[-1] = 0x0D
    body = blast(dbc[header_len + 4 :], max_size - header_len)
    return bytes(header) + body


def stream_dbf(dbf, chunk_size=30000):
    """Fetches records in parquet chunks to preserve memory"""
    data = []
    i = 0
    for records in dbf:
        data.append(records)
        i += 1
        if i == chunk_size:
            yield data
            data = []
            i = 0
    else:
        yield data


def decode_column(value):
    """
    Decodes binary data to str
    """
    if isinstance(value, bytes):
        return value.decode(encoding="iso-8859-1").replace("\x00", "")

    if isinstance(value, str):
        return str(value).replace("\x00", "")

    return value


def dbf_to_parquet(dbf: str, _pbar=None) -> str:
    """
    Parses DBF file into parquet to preserve memory
    """
    path = Path(dbf)

    if path.suffix.lower() != ".dbf":
        raise ValueError(f"Not a DBF file: {path}")

    parquet = path.with_suffix(".parquet")

    approx_final_size = (
        os.path.getsize(path) / 200
    )  # TODO: not best approx size
    if _pbar:
        _pbar.unit = "B"
        _pbar.unit_scale = True
        _pbar.reset(total=approx_final_size)
        _pbar.set_description(f"{parquet.name}")

    if parquet.exists():
        if _pbar:
            _pbar.update(approx_final_size - _pbar.n)
        return str(parquet)

    parquet.absolute().mkdir()

    try:
        chunk_size = 30_000
        for chunk in stream_dbf(
            DBF(path, encoding="iso-8859-1", raw=True), chunk_size
        ):
            if _pbar:
                _pbar.update(chunk_size)

            chunk_df = pd.DataFrame(chunk)
            table = pa.Table.from_pandas(chunk_df.map(decode_column))
            pq.write_to_dataset(table, root_path=str(parquet))
    except struct.error as err:
        if _pbar:
            _pbar.close()
        Path(path).unlink()
        parquet.rmdir()
        raise err

    if _pbar:
        _pbar.update(approx_final_size - _pbar.n)

    path.unlink()

    return str(parquet)


def parse_dftypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse DataFrame values, cleaning blank spaces if needed
    and converting dtypes into correct types.
    """

    def map_column_func(column_names: list[str], func):
        # Maps a function to each value in each column
        columns = [c for c in df.columns if c in column_names]
        df[columns] = df[columns].map(func)

    def str_to_int(string: str):
        # If removing spaces, all characters are int,
        # return int(value). @warning it removes in between
        # spaces as well
        if str(string).replace(" ", "").isnumeric():
            return int(string.replace(" ", ""))
        return string

    def str_to_date(string: str):
        if isinstance(string, str):
            try:
                return datetime.strptime(string, "%Y%m%d").date()
            except ValueError:
                # Ignore errors, bad value
                return string
        return string

    map_column_func(["DT_NOTIFIC", "DT_SIN_PRI"], str_to_date)
    map_column_func(["CODMUNRES", "SEXO"], str_to_int)

    df = df.map(
        lambda x: "" if str(x).isspace() else x
    )  # Remove all space values

    df = df.convert_dtypes()
    return df

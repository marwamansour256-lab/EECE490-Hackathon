"""Validate legacy byte hashes across Git's LF/CRLF text checkout conversion.

Only line-ending conversions are accepted. Other byte changes still fail. Raw
hashes remain in the original run records; portable hashes explicitly use LF.
Binary files are never normalized.
"""
import hashlib
from pathlib import Path

TEXT_SUFFIXES={".csv",".json",".js",".py",".md",".html",".txt"}


def portable_sha256(path):
    path=Path(path)
    data=path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data=data.replace(b"\r\n",b"\n")
    return hashlib.sha256(data).hexdigest()


def matches_recorded_sha256(path,expected):
    path=Path(path)
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()==expected:
        return True
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    lf=data.replace(b"\r\n",b"\n")
    return expected in {hashlib.sha256(lf).hexdigest(),hashlib.sha256(lf.replace(b"\n",b"\r\n")).hexdigest()}

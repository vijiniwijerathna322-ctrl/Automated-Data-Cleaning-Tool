"""File input/output. Depends on ``core`` only for typed errors and the report model."""

from datacleaner.io.formats import FileFormat, detect_format
from datacleaner.io.readers import read_bytes, read_path
from datacleaner.io.writers import to_bytes, write_path, write_report

__all__ = [
    "FileFormat",
    "detect_format",
    "read_bytes",
    "read_path",
    "to_bytes",
    "write_path",
    "write_report",
]

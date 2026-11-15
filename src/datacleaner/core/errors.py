"""Typed exceptions. The API layer maps these to HTTP status codes."""


class DataCleanerError(Exception):
    """Base class for every error raised deliberately by this package."""


class InvalidInputError(DataCleanerError):
    """The supplied dataset cannot be processed (empty, unreadable, too large...)."""


class InvalidConfigError(DataCleanerError):
    """The pipeline configuration is inconsistent with the dataset or itself."""

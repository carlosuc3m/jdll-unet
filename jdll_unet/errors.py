"""Typed exceptions for stable callers and clearer Appose error handling."""


class JdllUnetError(Exception):
    """Base class for package-specific failures."""


class ConfigError(ValueError, JdllUnetError):
    """Raised when a training or inference config is invalid."""


class DatasetError(ValueError, JdllUnetError):
    """Raised when a dataset layout or image/mask pair cannot be used."""


class DataFormatError(DatasetError):
    """Raised when an image or mask file has an unsupported format."""


class TaskDetectionError(ValueError, JdllUnetError):
    """Raised when task inference cannot produce a trainable task."""


class ModelLoadError(ValueError, JdllUnetError):
    """Raised when a model folder or checkpoint cannot be loaded safely."""


class InferenceError(ValueError, JdllUnetError):
    """Raised when inference inputs or options are invalid."""

class InterpolationError(Exception):
    """Base class for interpolation errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

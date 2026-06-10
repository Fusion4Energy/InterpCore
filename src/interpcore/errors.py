class InterpolationError(Exception):
    """Base class for interpolation errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class IncompatibleResultsError(Exception):
    """Raised when a request for a result that cannot be computed is asked"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

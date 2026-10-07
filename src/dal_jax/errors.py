"""Exceptions named after DAL's error codes.

DAL reports failures as ``"<Code>: <detail>"`` messages.  Each code here is a
subclass of :class:`DalError` carrying the same ``code`` so callers can catch a
specific failure (``except InvalidSetting``) or any DAL error at once.
"""


class DalError(Exception):
    """Base class; ``str(error)`` is ``"<code>: <detail>"`` as in DAL."""

    code = "DalError"

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        super().__init__(f"{self.code}: {detail}" if detail else self.code)


class InvalidSetting(DalError):
    code = "InvalidSetting"


class InvalidSmoothing(InvalidSetting):
    code = "InvalidSmoothing"


class ReservedIdentifier(DalError):
    code = "ReservedIdentifier"


class InvalidPathCount(DalError):
    code = "InvalidPathCount"


class InvalidModelParameter(DalError):
    code = "InvalidModelParameter"


class InvalidModelTimeline(DalError):
    code = "InvalidModelTimeline"


class InvalidBrownianBridge(DalError):
    code = "InvalidBrownianBridge"


class UnsupportedBrownianBridge(DalError):
    code = "UnsupportedBrownianBridge"


class UnsupportedModelObservation(DalError):
    code = "UnsupportedModelObservation"


class InvalidPayoff(DalError):
    code = "InvalidPayoff"


class InvalidRandomSequence(DalError):
    code = "InvalidRandomSequence"

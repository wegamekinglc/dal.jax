"""Exceptions named after DAL's error codes."""

import re
from types import MappingProxyType


class DalError(Exception):
    """Base class; ``str(error)`` is ``"<code>: <detail>"`` as in DAL."""

    code = "DalError"

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        super().__init__(f"{self.code}: {detail}" if detail else self.code)


class ScriptError(DalError):
    """Any failure DAL raises as ``ScriptError_``."""

    code = "ScriptError"


class InvalidSetting(ScriptError):
    code = "InvalidSetting"


class InvalidSmoothing(InvalidSetting):
    code = "InvalidSmoothing"


class InvalidPathCount(ScriptError):
    code = "InvalidPathCount"


class UnsupportedExecutionMode(ScriptError):
    code = "UnsupportedExecutionMode"


class InvalidPayoff(ScriptError):
    code = "InvalidPayoff"


class UnsupportedBrownianBridge(ScriptError):
    code = "UnsupportedBrownianBridge"


class InvalidScript(ScriptError):
    code = "InvalidScript"


class InvalidScriptStructure(ScriptError):
    code = "InvalidScriptStructure"


class ReservedIdentifier(ScriptError):
    code = "ReservedIdentifier"


class InvalidIndex(ScriptError):
    code = "InvalidIndex"


class UnknownIndex(ScriptError):
    code = "UnknownIndex"


class InvalidFixingDate(ScriptError):
    code = "InvalidFixingDate"


class InvalidPaymentDate(ScriptError):
    code = "InvalidPaymentDate"


class InvalidAssignmentTarget(ScriptError):
    code = "InvalidAssignmentTarget"


class InvalidPaymentTarget(ScriptError):
    code = "InvalidPaymentTarget"


class DuplicateElse(ScriptError):
    code = "DuplicateElse"


class InvalidFor(ScriptError):
    code = "InvalidFor"


class InvalidVectorDefinition(ScriptError):
    code = "InvalidVectorDefinition"


class InvalidVectorEntry(ScriptError):
    code = "InvalidVectorEntry"


class InvalidVectorReduction(ScriptError):
    code = "InvalidVectorReduction"


class InvalidVectorAppend(ScriptError):
    code = "InvalidVectorAppend"


class ImmutableVector(ScriptError):
    code = "ImmutableVector"


class VectorNameConflict(ScriptError):
    code = "VectorNameConflict"


class VectorIndexOutOfRange(ScriptError):
    code = "VectorIndexOutOfRange"


class EmptyVectorReduction(ScriptError):
    code = "EmptyVectorReduction"


class DuplicateExercise(ScriptError):
    code = "DuplicateExercise"


class UnsupportedExerciseNesting(ScriptError):
    code = "UnsupportedExerciseNesting"


class InvalidExerciseCondition(ScriptError):
    code = "InvalidExerciseCondition"


class UnsupportedExercisePayoff(ScriptError):
    code = "UnsupportedExercisePayoff"


class PreparationRequired(ScriptError):
    code = "PreparationRequired"


class UnboundHistoricalSpot(ScriptError):
    code = "UnboundHistoricalSpot"


class LookAheadObservation(ScriptError):
    code = "LookAheadObservation"


class MissingFixing(ScriptError):
    code = "MissingFixing"


class MissingDefaultIndex(ScriptError):
    code = "MissingDefaultIndex"


class InvalidFixingSnapshot(ScriptError):
    code = "InvalidFixingSnapshot"


class InvalidFixing(ScriptError):
    code = "InvalidFixing"


class MultipleModelIndices(ScriptError):
    code = "MultipleModelIndices"


class UnsupportedHistoricalIndex(ScriptError):
    code = "UnsupportedHistoricalIndex"


class UnsupportedDelayedPayment(ScriptError):
    code = "UnsupportedDelayedPayment"


class UnsettledDelayedPayment(ScriptError):
    code = "UnsettledDelayedPayment"


class DebugSchemaUnsupported(ScriptError):
    code = "DebugSchemaUnsupported"


class InvalidModelParameter(DalError):
    code = "InvalidModelParameter"


class InvalidModelTimeline(DalError):
    code = "InvalidModelTimeline"


class InvalidBrownianBridge(DalError):
    code = "InvalidBrownianBridge"


class UnsupportedModelObservation(DalError):
    code = "UnsupportedModelObservation"


class InvalidModelIndex(DalError):
    code = "InvalidModelIndex"


class DuplicateModelIndex(DalError):
    code = "DuplicateModelIndex"


class InvalidCorrelation(DalError):
    code = "InvalidCorrelation"


class InvalidLocalVolSurface(DalError):
    code = "InvalidLocalVolSurface"


class InvalidRandomSequence(DalError):
    code = "InvalidRandomSequence"


class InvalidDate(DalError):
    """Date construction or parsing outside DAL's supported range or formats."""

    code = "InvalidDate"


_CODE_PREFIX = re.compile(r"([A-Z][A-Za-z]+): ?")
_SCRIPT_CODES = MappingProxyType(
    {
        cls.code: cls
        for cls in (
            InvalidSetting,
            InvalidSmoothing,
            InvalidPathCount,
            UnsupportedExecutionMode,
            InvalidPayoff,
            UnsupportedBrownianBridge,
            InvalidScript,
            InvalidScriptStructure,
            ReservedIdentifier,
            InvalidIndex,
            UnknownIndex,
            InvalidFixingDate,
            InvalidPaymentDate,
            InvalidAssignmentTarget,
            InvalidPaymentTarget,
            DuplicateElse,
            InvalidFor,
            InvalidVectorDefinition,
            InvalidVectorEntry,
            InvalidVectorReduction,
            InvalidVectorAppend,
            ImmutableVector,
            VectorNameConflict,
            VectorIndexOutOfRange,
            EmptyVectorReduction,
            DuplicateExercise,
            UnsupportedExerciseNesting,
            InvalidExerciseCondition,
            UnsupportedExercisePayoff,
            PreparationRequired,
            UnboundHistoricalSpot,
            LookAheadObservation,
            MissingFixing,
            MissingDefaultIndex,
            InvalidFixingSnapshot,
            InvalidFixing,
            MultipleModelIndices,
            UnsupportedHistoricalIndex,
            UnsupportedDelayedPayment,
            UnsettledDelayedPayment,
            UnsupportedModelObservation,
            DebugSchemaUnsupported,
        )
    }
)


def script_error(message: str) -> ScriptError:
    """The :class:`ScriptError` subclass named by the message prefix; ``str()`` is ``message``."""
    match = _CODE_PREFIX.match(message)
    cls = _SCRIPT_CODES.get(match.group(1), ScriptError) if match else ScriptError
    error = cls.__new__(cls)
    Exception.__init__(error, message)
    error.detail = message[match.end() :] if match and cls is not ScriptError else message
    if cls is ScriptError and match:
        error.code = match.group(1)
    return error

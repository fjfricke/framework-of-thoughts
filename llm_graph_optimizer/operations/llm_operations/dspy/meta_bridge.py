from abc import ABCMeta
from dspy import Module             # just to import ProgramMeta

# ProgramMeta is the metaclass of dspy.Module
ProgramMeta = Module.__class__

class OperationModuleMeta(ProgramMeta, ABCMeta):
    """Subclass of both ProgramMeta and ABCMeta to allow composition of dspy.Module and BaseLLMOperation."""
    pass
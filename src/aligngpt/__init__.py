"""Import-safe primitives for the AlignGPT platform."""

from aligngpt.config import PlatformConfig
from aligngpt.schemas import AlignmentRequest, AlignmentResponse, BenchmarkResult, SafetyFinding
from aligngpt.service import AlignmentService

__all__ = [
    "AlignmentRequest",
    "AlignmentResponse",
    "AlignmentService",
    "BenchmarkResult",
    "PlatformConfig",
    "SafetyFinding",
]

__version__ = "0.2.0"

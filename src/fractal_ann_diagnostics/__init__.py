"""fractal-ann-diagnostics — intrinsic-dimension descriptors for predicting ANN failure modes.

Public API:
    diagnose(vectors, workload="recall@10", sample_size=2000) -> DiagnosticResult
"""

from .descriptors import (  # noqa: F401
    correlation_dimension,
    hubness,
    lid_mle,
    multifractal_width,
)
from .diagnostic import DiagnosticResult, diagnose  # noqa: F401

__version__ = "0.1.0"
__author__ = "Dineshkumar Malempati Hari"
__orcid__ = "0009-0003-1036-9477"
__license__ = "MIT"

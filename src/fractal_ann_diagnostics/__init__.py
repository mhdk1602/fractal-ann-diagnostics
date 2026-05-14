"""fractal-ann-diagnostics — intrinsic-dimension descriptors for predicting ANN failure modes.

Public API (intended; partial in v0.0.1):
    diagnose(vectors, workload="recall@10") -> DiagnosticResult
"""

from .descriptors import (  # noqa: F401
    correlation_dimension,
    hubness,
    lid_mle,
    multifractal_width,
)

__version__ = "0.0.1"
__author__ = "Dineshkumar Malempati Hari"
__orcid__ = "0009-0003-1036-9477"
__license__ = "MIT"


def diagnose(vectors, workload: str = "recall@10"):
    """Top-level entry point. Not implemented in v0.0.1 scaffold.

    See diagnostic.py for the intended interface.
    """
    from .diagnostic import diagnose as _diagnose

    return _diagnose(vectors, workload=workload)

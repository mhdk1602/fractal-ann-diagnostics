"""Geometry-aware control for authorization-first vector retrieval.

Public API:
    diagnose(vectors, workload="recall@10", sample_size=2000) -> DiagnosticResult
    GovernedRetriever(vectors, policy, role).query(query, k=10) -> GovernedResult
"""

from .controller import (  # noqa: F401
    ControllerConfig,
    GovernedResult,
    GovernedRetriever,
    RuleController,
)
from .descriptors import (  # noqa: F401
    correlation_dimension,
    hubness,
    lid_mle,
    multifractal_width,
)
from .diagnostic import DiagnosticResult, diagnose  # noqa: F401
from .geometry import QueryGeometry, multiscale_lid_dispersion, query_geometry  # noqa: F401
from .policy import AuthorizationPolicy  # noqa: F401

__version__ = "0.2.0"
__author__ = "mhdk1602"
__orcid__ = "0009-0003-1036-9477"
__license__ = "MIT"

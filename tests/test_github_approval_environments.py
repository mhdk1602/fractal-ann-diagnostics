from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW_ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _jobs(workflow_name: str) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load((WORKFLOW_ROOT / workflow_name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    jobs = payload["jobs"]
    assert isinstance(jobs, dict)
    return jobs


@pytest.mark.parametrize(
    ("workflow_name", "gated_jobs"),
    (
        (
            "confirmatory-c0-evidence-release.yml",
            {"publish", "cleanup-draft"},
        ),
        ("confirmatory-registration-attestation.yml", {"attest"}),
        ("confirmatory-state-attestation.yml", {"attest"}),
        ("confirmatory-online-execution.yml", {"claim", "execute"}),
        ("confirmatory-label-release.yml", {"claim", "execute"}),
        ("confirmatory-analysis.yml", {"claim", "execute"}),
    ),
)
def test_production_workflow_jobs_use_only_the_confirmatory_environment(
    workflow_name: str,
    gated_jobs: set[str],
) -> None:
    jobs = _jobs(workflow_name)
    assert {name for name, job in jobs.items() if "environment" in job} == gated_jobs
    for job_name in gated_jobs:
        assert jobs[job_name]["environment"] == "confirmatory"


def test_rehearsal_workflow_gates_only_the_three_execution_jobs() -> None:
    jobs = _jobs("confirmatory-provider-rehearsal.yml")
    gated_jobs = {name for name, job in jobs.items() if "environment" in job}
    assert gated_jobs == {"execute_online", "execute_label_release", "execute_analysis"}
    for job_name in gated_jobs:
        assert jobs[job_name]["environment"] == "confirmatory-rehearsal"


def test_image_publication_environment_is_derived_only_from_mode() -> None:
    jobs = _jobs("confirmatory-image.yml")
    assert {name for name, job in jobs.items() if "environment" in job} == {
        "instantiate_production_controls",
        "publish",
    }
    assert jobs["instantiate_production_controls"]["environment"] == "confirmatory"
    assert jobs["publish"]["environment"] == {
        "name": "${{ inputs.mode == 'production' && 'confirmatory' || 'confirmatory-rehearsal' }}"
    }

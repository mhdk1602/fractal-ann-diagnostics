"""Shared immutable constants for provider host and phase-plan contracts."""

from __future__ import annotations

PHASE_HOST_TOOL_CONTRACT_SCHEMA = "fractal-phase-host-tool-contract-v1"
PHASE_HOST_PROBE_SCHEMA = "fractal-phase-host-probe-v1"
DOCKER_SERVER_PROBE_SCHEMA = "fractal-docker-server-probe-v1"

OFFICIAL_GH_VERSION = "2.96.0"
OFFICIAL_GH_OSX_ARM64_ARCHIVE_URI = (
    "https://github.com/cli/cli/releases/download/v2.96.0/gh_2.96.0_macOS_arm64.zip"
)
OFFICIAL_GH_OSX_ARM64_ARCHIVE_SHA256 = (
    "f23a0c37d963aacc3bed703ccbd59b41c5ca22101fab7f00eb2b7cad23aba463"
)
OFFICIAL_GH_OSX_ARM64_ARCHIVE_BYTE_COUNT = 13_950_131
OFFICIAL_GH_OSX_ARM64_BINARY_SHA256 = (
    "b1d6c442fde99ca27c04e1e74d624895abe37785f4a3e9e9b684bf7586ce4bc8"
)
OFFICIAL_ACTIONS_RUNNER_VERSION = "2.335.1"
OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_URI = (
    "https://github.com/actions/runner/releases/download/v2.335.1/"
    "actions-runner-osx-arm64-2.335.1.tar.gz"
)
OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256 = (
    "e1a9bc7a3661e06fa0b129d15c2064fe65dc81a431001d8958a9db1409b73769"
)
OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_BYTE_COUNT = 127_138_003
OFFICIAL_ACTIONS_RUNNER_LISTENER_SHA256 = (
    "57a04bccf7e22e6e9e0cf92c691a5a8b87c8cfa86535548f131f422d53a0a4df"
)
OFFICIAL_ACTIONS_RUNNER_LISTENER_DLL_SHA256 = (
    "a969651efdf3b35e905968f6434dad4adcd5fd07d3f20e43595840f075cd1b15"
)
OFFICIAL_ACTIONS_RUNNER_CONFIG_SHA256 = (
    "4ad01727c3f29a0b6473d625412af6bdefc6c077763a6410f359c764fc0b3ae8"
)
OFFICIAL_ACTIONS_RUNNER_RUN_SHA256 = (
    "b39d7e0ca921a3189f7fe4e0a2f686b46719d4ccc2647f156f14407ec4517e8f"
)
REGISTERED_DOCKER_CLIENT_VERSION = "28.3.2"
REGISTERED_DOCKER_CLIENT_BUILD = "578ccf6"
REGISTERED_DOCKER_CLIENT_SHA256 = "9614e706a1bd7a56eaf739e7cd8da760df5ea536f062f1ffef306920d199f63f"
SOURCE_BUILT_LINUX_ARM64_TLE_SHA256 = (
    "ca9d498b6a3c1ea8edff9ace7bf00eb0f90ce67166343161f9a53f21900a6ef5"
)
SOURCE_BUILT_LINUX_ARM64_TLE_BYTE_COUNT = 13_303_934
SOURCE_BUILT_LINUX_ARM64_TLE_SOURCE_COMMIT = "7b54141a9733fd6fa207587a11148280e6fb020d"
OFFICIAL_PYTHON_BUILD_STANDALONE_VERSION = "3.12.13"
OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_URI = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260510/"
    "cpython-3.12.13%2B20260510-aarch64-apple-darwin-install_only.tar.gz"
)
OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_SHA256 = (
    "5a30271f8d345a5b02b0c9e4e31e0f1e1455a8e4a04fba95cd9762472abc3b17"
)
OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_BYTE_COUNT = 25_102_827
OFFICIAL_PYTHON_BUILD_STANDALONE_BINARY_SHA256 = (
    "14b79bc842a2c806fc8dc6ab16b3b13fabb6b3043a6868ccee1b1170a19388b3"
)

PHASE_HOST_PROBE_FIELDS = frozenset(
    {
        "architecture",
        "kernel_release",
        "logical_cpu_count",
        "operating_system",
        "operating_system_version",
        "physical_memory_bytes",
        "schema_version",
    }
)
DOCKER_SERVER_PROBE_FIELDS = frozenset(
    {
        "architecture",
        "cpu_count",
        "engine_build",
        "engine_version",
        "kernel_version",
        "memory_bytes",
        "operating_system",
        "schema_version",
    }
)
PHASE_HOST_TOOL_CONTRACT_FIELDS = frozenset(
    {
        "controlled_root",
        "docker_client_build",
        "docker_client_version",
        "docker_executable",
        "docker_executable_sha256",
        "docker_resolved_executable",
        "docker_server_probe",
        "docker_server_probe_receipt_sha256",
        "gh_archive_byte_count",
        "gh_archive_sha256",
        "gh_archive_uri",
        "gh_executable",
        "gh_executable_sha256",
        "gh_version",
        "host_architecture",
        "host_operating_system",
        "host_probe",
        "host_probe_receipt_sha256",
        "python_archive_byte_count",
        "python_archive_sha256",
        "python_archive_uri",
        "python_executable",
        "python_executable_sha256",
        "python_version",
        "runner_archive_byte_count",
        "runner_archive_sha256",
        "runner_archive_uri",
        "runner_config_executable",
        "runner_config_sha256",
        "runner_disable_update",
        "runner_ephemeral",
        "runner_listener_dll",
        "runner_listener_dll_sha256",
        "runner_listener_executable",
        "runner_listener_sha256",
        "runner_run_executable",
        "runner_run_sha256",
        "runner_unattended",
        "runner_version",
        "schema_version",
        "venv_root",
        "venv_symlink_inventory_sha256",
        "venv_tree_sha256",
    }
)

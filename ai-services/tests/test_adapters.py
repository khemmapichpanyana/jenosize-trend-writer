"""Adapter discovery and the vLLM arguments built from it (no Modal needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.adapters import read_active, scan_adapters, vllm_lora_args, write_active


def _adapter(root: Path, version: str, complete: bool = True) -> None:
    folder = root / f"jeno-lora-{version}"
    folder.mkdir(parents=True)
    if complete:
        (folder / "adapter_config.json").write_text("{}")


def test_nothing_trained_means_base_model_only(tmp_path: Path) -> None:
    assert scan_adapters(tmp_path / "missing") == []
    assert vllm_lora_args(tmp_path) == []
    assert read_active(tmp_path) is None


def test_incomplete_adapters_are_never_served(tmp_path: Path) -> None:
    # A run that crashed mid-save leaves a directory vLLM cannot load.
    _adapter(tmp_path, "v1")
    _adapter(tmp_path, "v2", complete=False)
    assert [a["version"] for a in scan_adapters(tmp_path)] == ["v1"]


def test_versions_sort_numerically_and_newest_is_active_by_default(tmp_path: Path) -> None:
    for version in ("v2", "v10", "v1"):
        _adapter(tmp_path, version)
    assert [a["version"] for a in scan_adapters(tmp_path)] == ["v1", "v2", "v10"]
    assert read_active(tmp_path) == "v10"


def test_activation_moves_the_alias(tmp_path: Path) -> None:
    _adapter(tmp_path, "v1")
    _adapter(tmp_path, "v2")
    write_active(tmp_path, "v1")
    args = vllm_lora_args(tmp_path)
    assert args[0] == "--lora-modules"
    assert f"jeno-lora-v1={tmp_path}/jeno-lora-v1" in args
    assert f"jeno-lora-v2={tmp_path}/jeno-lora-v2" in args
    assert f"jeno-lora={tmp_path}/jeno-lora-v1" in args  # alias -> the chosen one


def test_cannot_activate_a_missing_adapter(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_active(tmp_path, "v3")


def test_a_stale_marker_falls_back_to_the_newest(tmp_path: Path) -> None:
    _adapter(tmp_path, "v1")
    (tmp_path / "ACTIVE_ADAPTER").write_text("v9\n")  # points at something deleted
    assert read_active(tmp_path) == "v1"

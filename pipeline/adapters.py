"""Trained LoRA adapters on the Modal volume: listing and choosing the live one.

Standard library only, on purpose: the vLLM serving image imports this module
at start-up to decide which adapters to register, and must not pull in the
app's dependencies.

Layout on the `jeno-models` volume (mounted at /models):

    /models/jeno-lora-v1/            adapter_config.json, adapter_model.safetensors,
                                     jeno_train_metrics.json
    /models/jeno-lora-v2/            ...
    /models/ACTIVE_ADAPTER           "v2"  — which version the `jeno-lora` alias serves

Every trained version is served under its own name (`jeno-lora-v1`, …) so eval
and A/B requests can pick one explicitly; the product API asks for `jeno-lora`
and gets whichever is active.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PREFIX = "jeno-lora-"
ALIAS = "jeno-lora"
ACTIVE_FILE = "ACTIVE_ADAPTER"
_VERSION = re.compile(r"^v(\d+)$")


def adapter_dir(models_dir: str | Path, version: str) -> Path:
    return Path(models_dir) / f"{PREFIX}{version}"


def scan_adapters(models_dir: str | Path) -> list[dict[str, Any]]:
    """Every complete adapter, oldest version first.

    "Complete" means adapter_config.json exists: a training run that crashed
    mid-save leaves a directory vLLM cannot load, and it must not be served.
    """
    root = Path(models_dir)
    if not root.is_dir():
        return []
    found = []
    for path in root.iterdir():
        if not path.name.startswith(PREFIX):
            continue
        version = path.name[len(PREFIX) :]
        if not _VERSION.match(version) or not (path / "adapter_config.json").is_file():
            continue
        metrics: dict[str, Any] = {}
        metrics_file = path / "jeno_train_metrics.json"
        if metrics_file.is_file():
            try:
                metrics = json.loads(metrics_file.read_text())
            except ValueError:
                metrics = {}
        found.append(
            {
                "version": version,
                "path": str(path),
                "served_as": f"{PREFIX}{version}",
                "metrics": metrics,
            }
        )
    return sorted(found, key=lambda a: int(_VERSION.match(a["version"]).group(1)))  # type: ignore[union-attr]


def read_active(models_dir: str | Path) -> str | None:
    """The chosen version, or the newest complete one if none was chosen."""
    adapters = scan_adapters(models_dir)
    marker = Path(models_dir) / ACTIVE_FILE
    if marker.is_file():
        chosen = marker.read_text().strip()
        if any(a["version"] == chosen for a in adapters):
            return chosen
    return adapters[-1]["version"] if adapters else None


def write_active(models_dir: str | Path, version: str) -> None:
    if not any(a["version"] == version for a in scan_adapters(models_dir)):
        raise FileNotFoundError(f"no complete adapter for {version} in {models_dir}")
    (Path(models_dir) / ACTIVE_FILE).write_text(version + "\n")


def vllm_lora_args(models_dir: str | Path) -> list[str]:
    """`--lora-modules` arguments: every version by name, plus the alias."""
    adapters = scan_adapters(models_dir)
    if not adapters:
        return []
    modules = [f"{a['served_as']}={a['path']}" for a in adapters]
    active = read_active(models_dir)
    active_path = next(a["path"] for a in adapters if a["version"] == active)
    modules.append(f"{ALIAS}={active_path}")
    return ["--lora-modules", *modules]

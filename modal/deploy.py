"""Single deploy entrypoint for everything on Modal.

    make deploy-modal          # = modal deploy modal/deploy.py

Every Modal function in this repo registers on the same App (`common.app`), and
a deploy publishes the functions that the entrypoint imported. Deploying one
file at a time would publish only that file's functions, so there is exactly one
thing to deploy, and it imports all of them.
"""

from __future__ import annotations

import eval as _eval  # noqa: F401  base vs fine-tuned evaluation
import jobs as _jobs  # noqa: F401  jobs API + workers
import serve as _serve  # noqa: F401  vLLM server (base + LoRA)
import train as _train  # noqa: F401  QLoRA training

from common import app

__all__ = ["app"]

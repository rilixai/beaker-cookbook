"""Prepare the pinned public AppWorld assets in local and hosted evaluators."""

from __future__ import annotations

import fcntl
import hashlib
import tempfile
from pathlib import Path


COMMIT = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
APPS_SHA256 = "88d21fc526c1655bb3eee4adfca78ccac793921e4506f28f734ecdb19af77a62"
PROJECT = Path(__file__).resolve().parents[1]


def prepare_appworld() -> Path:
    import appworld
    import requests
    from appworld.common.constants import DATA_VERSION, PASSWORD, SALT
    from appworld.common.crypto import unpack_bundle
    from appworld.download import download_data

    cache = Path(tempfile.gettempdir()) / f"beaker-appworld-{COMMIT}"
    cache.mkdir(exist_ok=True)
    with (cache / "setup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        package = Path(appworld.__file__).parent
        if not (package / "apps" / "spotify" / "models.py").exists():
            bundle = cache / "apps.bundle"
            if not bundle.exists():
                response = requests.get(
                    f"https://media.githubusercontent.com/media/StonyBrookNLP/appworld/{COMMIT}/src/appworld/.source/apps.bundle",
                    timeout=120,
                )
                response.raise_for_status()
                if hashlib.sha256(response.content).hexdigest() != APPS_SHA256:
                    raise ValueError("Pinned AppWorld apps bundle checksum mismatch")
                bundle.write_bytes(response.content)
            if hashlib.sha256(bundle.read_bytes()).hexdigest() != APPS_SHA256:
                raise ValueError("Cached AppWorld apps bundle checksum mismatch")
            # Only the application bundle is required; do not unpack upstream tests.
            unpack_bundle(str(bundle), str(package), PASSWORD, SALT)
        root = PROJECT if (PROJECT / "data" / "datasets" / "train.txt").exists() else cache
        appworld.update_root(str(root))
        if not (root / "data" / "datasets" / "train.txt").exists():
            download_data(DATA_VERSION, mode="minimal")
        if not (root / "data" / "datasets" / "train.txt").exists():
            raise RuntimeError("AppWorld labeled data download did not complete")
        if (root / "data/version.txt").read_text().strip() != DATA_VERSION:
            raise ValueError("AppWorld data version does not match the pinned evaluator")
        return root

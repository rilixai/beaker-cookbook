"""Prepare the pinned upstream simulator and public benchmark data."""

import hashlib
import tempfile
from pathlib import Path

from filelock import FileLock


APPWORLD_COMMIT = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"


def prepare_resources() -> Path:
    import appworld
    import requests
    from appworld.common.constants import DATA_VERSION, PASSWORD, SALT
    from appworld.common.crypto import unpack_bundle
    from appworld.common.path_store import path_store
    from appworld.download import download_data

    cache = Path(tempfile.gettempdir()) / f"beaker-appworld-{APPWORLD_COMMIT}"
    cache.mkdir(exist_ok=True)
    package = Path(appworld.__file__).parent
    with FileLock(str(cache / "install.lock")):
        if not (package / "apps" / "amazon" / "models.py").is_file():
            bundle = package / ".source" / "apps.bundle"
            # Hosted pip installs may retain the upstream Git LFS pointer.
            # Resolve only this pinned object and verify its LFS content hash.
            if bundle.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
                pointer = dict(line.split(" ", 1) for line in bundle.read_text().splitlines())
                url = (
                    "https://media.githubusercontent.com/media/StonyBrookNLP/appworld/"
                    f"{APPWORLD_COMMIT}/src/appworld/.source/apps.bundle"
                )
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                content = response.content
                if len(content) != int(pointer["size"]) or hashlib.sha256(content).hexdigest() != pointer[
                    "oid"
                ].removeprefix("sha256:"):
                    raise ValueError("Upstream AppWorld bundle failed its Git LFS integrity check")
                bundle = cache / "apps.bundle"
                bundle.write_bytes(content)
            unpack_bundle(str(bundle), str(package), password=PASSWORD, salt=SALT)

    local_root = Path(__file__).resolve().parent.parent
    if (local_root / "data" / "version.txt").is_file():
        data_root = local_root
    else:
        data_root = cache
        with FileLock(str(cache / "data.lock")):
            if not (cache / "data" / "version.txt").is_file():
                previous_root = path_store.root
                try:
                    path_store.update_root(str(cache))
                    download_data(version=DATA_VERSION)
                finally:
                    path_store.update_root(previous_root)
    if (data_root / "data" / "version.txt").read_text().strip() != DATA_VERSION:
        raise ValueError("AppWorld data does not match the pinned simulator version")
    return data_root

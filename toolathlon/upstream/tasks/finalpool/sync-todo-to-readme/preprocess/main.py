import os
import json
import shutil
import asyncio
from argparse import ArgumentParser
from pathlib import Path
import time

from utils.general.helper import fork_repo
from configs.token_key_session import all_token_key_session
from utils.app_specific.github.api import (
    github_get_login, github_create_issue, github_delete_repo, 
    github_create_user_repo, github_get_latest_commit, github_get_repo
)
from utils.app_specific.github.git_ops import git_mirror_clone, git_mirror_push
from utils.app_specific.github.repo_ops import update_file_content
from utils.app_specific.huggingface.datasets import (
    hf_get_namespace, hf_create_dataset, hf_delete_dataset, hf_upload_file
)

GITHUB_REPO_NAME = "LUFFY"
SOURCE_REPO_NAME = f"Toolathlon-Archive/{GITHUB_REPO_NAME}"

async def main():
    parser = ArgumentParser()
    parser.add_argument("--agent_workspace", required=False)
    parser.add_argument("--token_path", required=False, default="configs/token_key_session.py")
    parser.add_argument("--launch_time", required=False)
    args = parser.parse_args()

    github_token = all_token_key_session.github_token

    # # Resolve dynamic namespaces/logins
    github_owner = github_get_login(github_token)
    github_repo_full = f"{github_owner}/{GITHUB_REPO_NAME}"

    # 2) Check if target repo exists, delete it if so
    # (Skip fork step to avoid GitHub's fork chain issue)
    try:
        if github_get_repo(github_token, github_owner, GITHUB_REPO_NAME):
            print(f"Target repo {github_repo_full} exists, deleting it...")
            github_delete_repo(github_token, github_owner, GITHUB_REPO_NAME)
            time.sleep(2)
    except RuntimeError as e:
        # 404 means repo doesn't exist, which is fine
        if "404" not in str(e):
            raise

    # 2.6) Mirror clone source repo directly (not the fork)
    tmpdir = Path(os.path.dirname(__file__)) / ".." / "tmp"
    tmpdir.mkdir(exist_ok=True)
    local_mirror_dir = tmpdir / f"{GITHUB_REPO_NAME}.git"
    print(f"Mirror cloning {SOURCE_REPO_NAME}...")
    await git_mirror_clone(github_token, SOURCE_REPO_NAME, str(local_mirror_dir))

    # 2.8) Create a new independent repo with the same name
    print(f"Creating new independent repo {github_repo_full}...")
    github_create_user_repo(github_token, GITHUB_REPO_NAME, private=False)

    # 2.9) Push mirror to the new repo
    print(f"Pushing mirror to {github_repo_full}...")
    await git_mirror_push(github_token, str(local_mirror_dir), github_repo_full)

    # Cleanup
    shutil.rmtree(tmpdir)

    with open(os.path.join(args.agent_workspace, ".github_token"), "w", encoding="utf-8") as f:
        f.write(github_token)

if __name__ == "__main__":
    asyncio.run(main())
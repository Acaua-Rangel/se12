"""Adapter: WeightSource over a Hugging Face Hub model repo.

Downloads one shard at a time into a temporary directory, reads it, then
deletes the directory before moving to the next shard — peak extra disk is
one shard, not the whole model (T-017 notes; verified against a real public
repo: `hf_hub_download(..., local_dir=...)` writes straight into that
directory, it does not also populate the global blob cache). Gated
checkpoints (Gemma, Llama) need `HF_TOKEN` in the environment (principle
P-002) and the license accepted on the Hub.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import typing

import huggingface_hub

from lws.adapters.safetensors.weight_source import _read_shard
from lws.application.ports.codec.weight_source import SourceTensor

SHARD_SUFFIX = ".safetensors"


class HuggingFaceShardStreamSource:
    """Streams a Hub repo's tensors, one downloaded-then-deleted shard at a time."""

    def __init__(self, repo_id: str, revision: str) -> None:
        self.repo_id = repo_id
        self.revision = revision

    def tensors(self) -> typing.Iterator[SourceTensor]:
        repo_id = self.repo_id
        revision = self.revision
        for filename in _shard_filenames(repo_id, revision):
            yield from _download_read_delete(repo_id, revision, filename)


def resolve_revision(repo_id: str) -> str:  # calisthenics: allow 3 — plain repo id/commit-hash strings, matches ModelIdentity's own field
    """The repo's current commit hash — pinned once, then used for every
    download of that model, and recorded in the survey report (AC-034)."""
    token = os.environ.get("HF_TOKEN")
    api = huggingface_hub.HfApi(token=token)
    repo_info = api.model_info(repo_id)
    return repo_info.sha


def _shard_filenames(repo_id: str, revision: str) -> list[str]:
    token = os.environ.get("HF_TOKEN")
    files = huggingface_hub.list_repo_files(repo_id, revision=revision, token=token)
    return sorted(name for name in files if name.endswith(SHARD_SUFFIX))


def _download_read_delete(repo_id: str, revision: str, filename: str) -> typing.Iterator[SourceTensor]:
    with tempfile.TemporaryDirectory() as temp_directory:
        downloaded_path = _download_one(repo_id, revision, filename, temp_directory)
        yield from _read_shard(downloaded_path)


def _download_one(repo_id: str, revision: str, filename: str, temp_directory: str) -> pathlib.Path:
    token = os.environ.get("HF_TOKEN")
    downloaded = huggingface_hub.hf_hub_download(repo_id, filename, revision=revision, token=token, local_dir=temp_directory)
    return pathlib.Path(downloaded)

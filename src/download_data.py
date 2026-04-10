
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ZurichNLP/ArgMining-2026-UZH-Shared-Task",
    repo_type="dataset",
    token="ENTER YOUR TOKEN HERE",
    local_dir="../Data",
    )

"""Download only a pinned public generic checkpoint, never a ship dataset."""
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import snapshot_download

REPO="Qwen/Qwen3-Reranker-4B"
REVISION="22e683669bc0f0bd69640a1354a6d0aebcfeede5"
MODEL=Path("D:/KSH/hf_models/Qwen__Qwen3-Reranker-4B")


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def main():
    start=time.perf_counter()
    if (MODEL/"codex_revision.json").exists():
        assert json.loads((MODEL/"codex_revision.json").read_text())["revision"]==REVISION
    snapshot_download(REPO,revision=REVISION,local_dir=MODEL,max_workers=2,
                      allow_patterns=["config.json","generation_config.json","model*.safetensors","model.safetensors.index.json",
                                      "tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","special_tokens_map.json",
                                      "README.md","LICENSE*"])
    weights=sorted(MODEL.glob("model*.safetensors"));assert len(weights)==2
    config=json.loads((MODEL/"config.json").read_text());assert config["model_type"]=="qwen3"
    result=dict(repo=REPO,revision=REVISION,license="apache-2.0",source="Official generic pretrained reranker only; no external ship data",
                files_sha256={p.name:sha(p) for p in MODEL.iterdir() if p.is_file()},
                weight_bytes=sum(p.stat().st_size for p in weights),download_and_hash_seconds=time.perf_counter()-start,
                trust_remote_code=False,new_training=False,actual_competition_submission=False)
    (MODEL/"codex_revision.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="files_sha256"},indent=2),flush=True)


if __name__=="__main__":main()

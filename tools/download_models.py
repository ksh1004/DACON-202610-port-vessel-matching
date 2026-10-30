"""LLM 후보 가중치를 D 드라이브에 내려받는다 (추론 서버는 인터넷이 차단되므로 model/ 에 넣을 원본).

    python tools/download_models.py            # 전체 후보
    python tools/download_models.py Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ROOT = Path("D:/KSH/hf_models")
CANDIDATES = [
    "Helsinki-NLP/opus-mt-ko-en",
    "Qwen/Qwen3-1.7B",
    "kakaocorp/kanana-1.5-2.1b-instruct-2505",
    "K-intelligence/Midm-2.0-Mini-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]
PATTERNS = ["*.json", "*.safetensors", "pytorch_model.bin", "*.model", "*.txt", "*.py", "*.jinja", "*.spm", "*.vocab", "LICENSE*"]


def local_dir(repo_id: str) -> Path:
    return MODEL_ROOT / repo_id.replace("/", "__")


def main() -> None:
    repos = sys.argv[1:] or CANDIDATES
    for repo in repos:
        target = local_dir(repo)
        print(f"[download] {repo} -> {target}", flush=True)
        snapshot_download(repo, local_dir=target, allow_patterns=PATTERNS, max_workers=8)
        size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 2**30
        print(f"[done] {repo} {size:.2f} GB", flush=True)


if __name__ == "__main__":
    main()

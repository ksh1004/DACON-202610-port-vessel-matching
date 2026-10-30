"""로컬 가상환경이 DACON 평가 서버와 같은 버전인지 점검한다."""

from importlib.metadata import version
import sys

EXPECTED = {
    "torch": "2.7.1+cu128",
    "pandas": "2.0.3",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "scikit-learn": "1.5.2",
    "joblib": "1.5.2",
    "threadpoolctl": "3.7.0",
    "lightgbm": "4.6.0",
    "catboost": "1.2.8",
    "rapidfuzz": "3.14.3",
    "Unidecode": "1.4.0",
    "transformers": "4.57.6",
    "accelerate": "1.9.0",
    "safetensors": "0.6.2",
    "sentencepiece": "0.2.0",
    "tqdm": "4.66.5",
}


def main() -> int:
    ok = sys.version_info[:2] == (3, 11)
    print(f"python {sys.version.split()[0]} (server 3.11.17) {'OK' if ok else 'MISMATCH'}")
    for name, expected in EXPECTED.items():
        try:
            found = version(name)
        except Exception:
            found = "not installed"
        match = found == expected
        ok &= match
        print(f"{name:15s} {found:15s} {'OK' if match else 'MISMATCH (server ' + expected + ')'}")

    import torch

    cuda = torch.cuda.is_available()
    print(f"cuda available: {cuda}", end="")
    if cuda:
        x = torch.randn(256, 256, device="cuda")
        _ = (x @ x).sum().item()
        print(f" / {torch.cuda.get_device_name(0)} / matmul OK")
    else:
        print()
    print("ALL OK" if ok else "SOME MISMATCHES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Bundle what the Kaggle notebook needs into dist/minecraft-rag-kaggle.zip (~25 MB).

Upload the zip as a private Kaggle Dataset named `minecraft-rag`; Kaggle extracts it under
/kaggle/input/minecraft-rag/. The raw crawl (data/) is not needed - the built index is included.
Run from the repo root:  python kaggle/package.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "minecraft-rag-kaggle.zip"
INCLUDE = [
    "mcrag/*.py",
    "eval/*.json",
    "index/*",
    "requirements.txt",
    "eval/results/_state.json",
]


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    files = sorted({p for pattern in INCLUDE for p in ROOT.glob(pattern) if p.is_file()})
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()
            if p.suffix in {".py", ".json", ".txt"}:
                data = data.replace(b"\r\n", b"\n")  # Linux line endings on Kaggle
            z.writestr(f"minecraft-rag/{rel}", data)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()

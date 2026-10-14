"""按文件白名单打包源码，并记录每个文件的 SHA-256。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out or root.with_suffix(".zip")
    excluded_names = {"实现与验证边界说明.md"}
    names = {
        "README.md",
        "REVIEW.md",
        "pyproject.toml",
        "requirements-lock-macos-arm64.txt",
        "CITATION.cff",
        "references.bib",
        "THIRD_PARTY.md",
        "LICENSE-NOTICE.md",
        ".gitignore",
    }
    files = [root / name for name in names if (root / name).is_file()]
    for directory, suffixes in {
        "src": {".py"},
        "tests": {".py"},
        "configs": {".json"},
        "docs": {".md", ".svg"},
        "tools": {".py"},
        "review": {".json"},
        ".github": {".yml", ".yaml"},
    }.items():
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and path.name not in excluded_names
            and path.suffix in suffixes
            and "__pycache__" not in path.parts
            and (
                directory != "review" or path.name in {"software_validation.json", "preflight.json"}
            )
        )
    files.sort()
    manifest = {
        "scope": "source_distribution",
        "includes_event_data": False,
        "includes_model_checkpoints": False,
        "empirical_validation_included": False,
        "sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    manifest_path = root / "SOURCE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files.append(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path(root.name) / path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("压缩包完整性校验失败")
        for relative, expected in manifest["sha256"].items():
            if hashlib.sha256(archive.read(f"{root.name}/{relative}")).hexdigest() != expected:
                raise RuntimeError(f"文件摘要不一致：{relative}")
    print(
        f"{output}；{len(files)} 个文件；SHA-256={hashlib.sha256(output.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    main()

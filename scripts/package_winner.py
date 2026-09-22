#!/usr/bin/env python3
"""Empacota código, ambiente, modelos e CSV vencedor com manifesto SHA-256.

Os dados ERA5 não entram no ZIP; devem ser obtidos pela competição conforme a
licença. O script não envia nada ao Kaggle ou ao patrocinador.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

from worcap_pipeline import ROOT, audit_submission, sha256_file


SOURCE_PATHS = [
    ROOT / "README.md",
    ROOT / "requirements-worcap.txt",
    ROOT / "scripts" / "worcap_pipeline.py",
    ROOT / "scripts" / "worcap_unet.py",
    ROOT / "scripts" / "package_winner.py",
    ROOT / "tests" / "test_worcap_pipeline.py",
    ROOT / "docs" / "ENTREGA_MODELO.md",
    ROOT / "docs" / "PROTOCOLO_EXPERIMENTOS.md",
    ROOT / "notebooks" / "worcap_final.ipynb",
    ROOT / "experiments" / "registry.csv",
]


def _git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        return completed.stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def package_winner(
    submission: Path,
    sample: Path,
    artifacts: Sequence[Path],
    output: Path,
) -> Path:
    audit = audit_submission(submission, sample)
    required = SOURCE_PATHS + [submission, *artifacts]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Arquivos ausentes: {missing}")
    entries: dict[str, dict[str, object]] = {}
    used_names: set[str] = set()
    archive_items: list[tuple[Path, str]] = []
    for path in required:
        resolved = path.resolve()
        try:
            archive_name = str(resolved.relative_to(ROOT.resolve()))
        except ValueError:
            archive_name = f"artifacts/{resolved.name}"
        if archive_name in used_names:
            raise ValueError(f"Nome duplicado no pacote: {archive_name}")
        used_names.add(archive_name)
        archive_items.append((resolved, archive_name))
        entries[archive_name] = {"sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}
    manifest = {
        "submission_audit": audit,
        "git": _git_state(),
        "files": entries,
        "data_not_included": "Use os 13 arquivos oficiais da competição Kaggle, sujeitos às regras/licença.",
        "training_entrypoint": "scripts/worcap_pipeline.py fit-final ou scripts/worcap_unet.py fit-final",
        "inference_entrypoint": "scripts/worcap_pipeline.py predict ou scripts/worcap_unet.py predict",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        manifest_path = Path(temporary) / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for source, archive_name in archive_items:
                archive.write(source, archive_name)
            archive.write(manifest_path, "MANIFEST.json")
    print(json.dumps({"package": str(output), "sha256": sha256_file(output), **manifest}, indent=2, ensure_ascii=False))
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--artifact", action="append", type=Path, default=[], help="Modelo/metadata/OOF/ambiente; repita")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    package_winner(args.submission, args.sample, args.artifact, args.output)


if __name__ == "__main__":
    main()

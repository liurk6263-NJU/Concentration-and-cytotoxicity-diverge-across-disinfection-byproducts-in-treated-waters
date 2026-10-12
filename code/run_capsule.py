#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def run_command(name: str, command: list[str], cwd: Path, log_dir: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log = completed.stdout or ""
    (log_dir / f"{name}.log").write_text(log, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}; see {log_dir / (name + '.log')}")
    return log


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    capsule_dir = code_dir.parent
    source_dir = code_dir / "analysis"
    data_dir = capsule_dir / "data" / "analysis"
    frozen_results_dir = capsule_dir / "results" / "analysis"
    output_dir = capsule_dir / "results" / "reproducible_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()
        elif old.is_dir():
            shutil.rmtree(old)

    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    workspace = Path(tempfile.mkdtemp(prefix="nature_water_repro_"))
    records: list[dict[str, object]] = []
    try:
        copy_tree(source_dir, workspace)
        copy_tree(data_dir, workspace)
        copy_tree(frozen_results_dir, workspace)
        protocol_script = code_dir / "audit_protocol.py"
        commands = [
            ("protocol_audit", [sys.executable, str(protocol_script), str(workspace)]),
            ("verification", [sys.executable, "verify.py"]),
            ("screening_selection", [sys.executable, "04_screening/code/selection.py"]),
            ("screening_domain", [sys.executable, "04_screening/code/domain.py"]),
            ("treated_water", [sys.executable, "02_treated_water/code/calculate.py"]),
            ("cellular_state", [sys.executable, "05_supplementary/S4_cellular_state/calculate.py"]),
        ]
        for name, command in commands:
            log = run_command(name, command, workspace, output_dir, environment)
            records.append({"name": name, "command": command, "status": "passed", "log": f"{name}.log"})
            if name == "verification":
                try:
                    verification = json.loads(log)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("verification.py did not emit JSON") from exc
                (output_dir / "verification.json").write_text(
                    json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
            elif name == "screening_selection":
                (output_dir / "screening_selection.json").write_text(log.strip() + "\n", encoding="utf-8")
            elif name == "protocol_audit":
                (output_dir / "protocol_audit.json").write_text(log, encoding="utf-8")

        generated = workspace / "02_treated_water" / "results" / "class_contributions.csv"
        if generated.exists():
            shutil.copy2(generated, output_dir / generated.name)
        for relative in (
            "01_models/figures/Figure2.png",
            "02_treated_water/Figure3.png",
            "03_mechanistic/Figure4.png",
            "04_screening/Figure5ab.png",
            "04_screening/Figure5c.png",
        ):
            source = workspace / relative
            if source.exists():
                shutil.copy2(source, output_dir / source.name)
        manifest = {
            "runner": "code/run_capsule.py",
            "workspace_reconstructed": True,
            "commands": records,
            "outputs": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "passed", "results": str(output_dir), "commands": len(records)}, ensure_ascii=False))
        return 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

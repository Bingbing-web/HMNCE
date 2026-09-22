"""Reproducible HMNCE experiment launcher.

This replaces the historical workflow of copying different model files over
``model/drae_model.py``.  Every experiment uses the same HMNCE implementation
and changes behavior only through explicit command-line flags.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROFILES = {
    "full": [],
    "w-o-me": ["--disable_me"],
    "w-o-hcl": ["--disable_hcl"],
    "w-o-srm": ["--disable_srm"],
    "w-o-me-hcl": ["--disable_me", "--disable_hcl"],
    "w-o-me-srm": ["--disable_me", "--disable_srm"],
    "w-o-hcl-srm": ["--disable_hcl", "--disable_srm"],
    "random-mining": ["--disable_hsm"],
    "top1": ["--disable_hsa"],
    "score-transe": ["--hmnce-score-func", "transe"],
    "score-complex": ["--hmnce-score-func", "complex"],
    "score-conve": ["--hmnce-score-func", "conve"],
    "score-tucker": ["--hmnce-score-func", "tucker"],
}

GROUPS = {
    "ablation": ["full", "w-o-me", "w-o-hcl", "w-o-srm", "w-o-me-hcl", "w-o-me-srm", "w-o-hcl-srm"],
    "hard-sample": ["full", "random-mining", "top1"],
    "scores": ["score-transe", "score-complex", "score-conve", "score-tucker"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch named HMNCE experiment profiles")
    parser.add_argument("profiles", nargs="+", help="Profile names or groups: ablation, hard-sample, scores")
    parser.add_argument("--data", default="hubeiquanbu_type5000")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def expand_profiles(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        names = GROUPS.get(value, [value])
        for name in names:
            if name not in PROFILES:
                raise ValueError(f"Unknown profile: {name}. Available: {', '.join(sorted(PROFILES))}")
            if name not in expanded:
                expanded.append(name)
    return expanded


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    for profile in expand_profiles(args.profiles):
        command = [
            sys.executable,
            str(project / "train.py"),
            "--model", "hmnce",
            "--name", f"hmnce_{args.data}_{profile}",
            "--data", args.data,
            "--data-root", str(args.data_root.resolve()),
            "--output-root", str(args.output_root.resolve()),
            "--gpu", str(args.gpu),
            "--epoch", str(args.epoch),
            "--batch", str(args.batch),
            *PROFILES[profile],
            *args.extra,
        ]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=project)


if __name__ == "__main__":
    main()

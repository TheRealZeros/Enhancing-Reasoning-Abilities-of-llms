#!/usr/bin/env python3
"""Compatibility entrypoint for Phase 4b attention heatmaps."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_4b_attention_visualisation.attention_heatmaps import main


if __name__ == "__main__":
    main()

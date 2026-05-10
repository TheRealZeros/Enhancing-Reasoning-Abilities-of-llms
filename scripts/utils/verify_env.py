import argparse
import os
import sys
import platform
from pathlib import Path

# ---- Parse args before any checks run ----
_parser = argparse.ArgumentParser(
    description="Verify that the environment is ready to run experiments."
)
_parser.add_argument(
    "--model", type=str, default="EleutherAI/pythia-2.8b",
    help="HuggingFace model name to use for the TransformerLens load test "
         "(default: EleutherAI/pythia-2.8b)"
)
_parser.add_argument(
    "--skip-model-load", action="store_true",
    help="Skip the model load test (faster; useful when only checking packages)"
)
_args = _parser.parse_args()

results = []


def check(name, fn):
    try:
        value = fn()
        results.append((name, True, value))
        return value
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}"))
        return None


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    # verify_env.py -> utils -> scripts -> repo root
    return current.parents[2]


def check_numpy_compat():
    import numpy as np

    version_str = np.__version__
    major_minor = tuple(int(x) for x in version_str.split(".")[:2])

    if major_minor >= (2, 0):
        raise RuntimeError(
            f"Incompatible numpy version for TransformerLens detected: {version_str}. "
            f"Expected numpy < 2."
        )
    return version_str


print_header("Environment Verification Report")

repo_root = find_repo_root()

check("Python executable", lambda: sys.executable)
check("Python version", lambda: sys.version.replace("\n", " "))
check("Platform", lambda: platform.platform())
check("Repository root", lambda: str(repo_root))
check("Current working directory", lambda: os.getcwd())

os.chdir(repo_root)
check("Changed working directory to repo root", lambda: os.getcwd())

def import_torch():
    import torch
    return torch

torch = check("Import torch", import_torch)

if torch is not None:
    check("Torch version", lambda: torch.__version__)
    check("Torch CUDA version", lambda: torch.version.cuda)
    cuda_available = check("CUDA available", lambda: torch.cuda.is_available())

    if cuda_available:
        check("CUDA device count", lambda: torch.cuda.device_count())
        check("CUDA device name", lambda: torch.cuda.get_device_name(0))
        check(
            "CUDA total memory (GB)",
            lambda: round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        )
    else:
        results.append(("CUDA note", True, "CUDA not available; model will load on CPU if possible"))

check("Import numpy", check_numpy_compat)
check("Import transformers", lambda: __import__("transformers").__version__)
check("Import transformer_lens", lambda: __import__("transformer_lens"))

def import_hooked_transformer():
    from transformer_lens import HookedTransformer
    return HookedTransformer

HookedTransformer = check("Import HookedTransformer", import_hooked_transformer)

required_dirs = [
    "dataset", "dataset/raw", "dataset/processed",
    "scripts", "results", "figures", "setup-env",
]
# Phase-specific output directories (created by scripts, checked here for completeness)
phase_dirs = [
    "results/phase_1_dataset",
    "results/phase_2_behaviour",
    "results/phase_3a_layer_patching",
    "results/phase_3b_component_patching",
    "results/phase_3c_cross_condition",
    "results/phase_4_logit_lens",
    "figures/phase_3a_layer_patching",
    "figures/phase_3b_component_patching",
    "figures/phase_4_logit_lens",
]
for d in required_dirs:
    check(f"Directory exists: {d}", lambda d=d: (repo_root / d).is_dir())
for d in phase_dirs:
    exists = (repo_root / d).is_dir()
    if exists:
        check(f"Phase dir exists: {d}", lambda d=d: True)
    else:
        # Phase dirs are created by scripts; note as info, not failure
        results.append((f"Phase dir missing (OK before first run): {d}", True,
                        "will be created by the corresponding script"))

if HookedTransformer is not None and torch is not None and not _args.skip_model_load:
    def load_model():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = HookedTransformer.from_pretrained(
            _args.model,
            device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        tokens = model.to_tokens("Hello world")
        logits = model(tokens)

        return f"Model {_args.model!r} loaded and forward pass succeeded on {device}"

    check(f"Load HookedTransformer model ({_args.model})", load_model)
elif _args.skip_model_load:
    results.append(("Model load test", True, "skipped via --skip-model-load"))

print_header("Summary")

passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)

for name, ok, value in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {value}")

print("\n" + "-" * 80)
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print("-" * 80)

if failed > 0:
    print("\nEnvironment verification failed.")
    sys.exit(1)

print("\nEnvironment verification passed.")
sys.exit(0)
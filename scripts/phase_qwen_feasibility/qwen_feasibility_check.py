#!/usr/bin/env python3
"""
qwen_feasibility_check.py — Qwen2.5-3B feasibility gate for the two-hop pipeline.

Runs a sequence of targeted checks to determine whether Qwen2.5-3B can be
dropped into the existing activation-patching pipeline WITHOUT silent failures.

Each check is rated: SAFE / RISKY / BLOCKED
  SAFE    — assumption holds; no code change needed
  RISKY   — assumption holds partially; behaviour should be verified at scale
  BLOCKED — assumption breaks; code change required before use

Usage:
  python scripts/phase_qwen_feasibility/qwen_feasibility_check.py
  python scripts/phase_qwen_feasibility/qwen_feasibility_check.py --model Qwen/Qwen2.5-3B
  python scripts/phase_qwen_feasibility/qwen_feasibility_check.py --device cpu
  python scripts/phase_qwen_feasibility/qwen_feasibility_check.py --skip-model-load
"""

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

SAFE    = "SAFE"
RISKY   = "RISKY"
BLOCKED = "BLOCKED"
SKIP    = "SKIP"
ERROR   = "ERROR"

@dataclass
class CheckResult:
    name: str
    status: str          # SAFE / RISKY / BLOCKED / SKIP / ERROR
    detail: str
    required_change: Optional[str] = None


results: List[CheckResult] = []


def record(name: str, status: str, detail: str, required_change: str = None):
    r = CheckResult(name=name, status=status, detail=detail,
                    required_change=required_change)
    results.append(r)
    marker = {"SAFE": "✓", "RISKY": "~", "BLOCKED": "✗",
               "SKIP": "-", "ERROR": "!"}.get(status, "?")
    print(f"  [{marker}] {name}: {detail}")
    if required_change:
        print(f"       CHANGE REQUIRED: {required_change}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_model_load(model_name: str, device: str) -> Optional[object]:
    """Check 1: Can TransformerLens load the model at all?"""
    print("\n[CHECK 1] Model loading")
    try:
        import torch
        from transformer_lens import HookedTransformer
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
        )
        record(
            "Model load",
            SAFE,
            f"{model_name} loaded on {device}. "
            f"n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}, "
            f"d_model={model.cfg.d_model}",
        )
        return model
    except Exception as e:
        record(
            "Model load",
            BLOCKED,
            f"Failed: {type(e).__name__}: {e}",
            required_change=(
                "Install or upgrade transformer_lens with Qwen2.5 support. "
                "See: pip install transformer_lens --upgrade"
            ),
        )
        return None


def check_tokenizer_basics(model):
    """Check 2: Tokenizer is present and has expected special tokens."""
    print("\n[CHECK 2] Tokenizer basics")
    tok = model.tokenizer
    if tok is None:
        record("Tokenizer present", BLOCKED, "model.tokenizer is None",
               required_change="Pass tokenizer explicitly or upgrade TransformerLens.")
        return

    record("Tokenizer present", SAFE, f"type={type(tok).__name__}")

    eos = tok.eos_token
    eos_id = tok.eos_token_id
    record("EOS token", SAFE if eos else BLOCKED,
           f"eos_token={eos!r}, id={eos_id}")

    bos = tok.bos_token
    record("BOS token", SAFE if bos else RISKY,
           f"bos_token={bos!r}, id={tok.bos_token_id}")

    pad = tok.pad_token
    record("PAD token",
           SAFE if pad else RISKY,
           f"pad_token={pad!r} (None means fallback to EOS, which may be multi-token)")


def check_eos_token_length(model):
    """Check 3: Is EOS exactly 1 token? The align_cells() padding logic assumes this."""
    print("\n[CHECK 3] EOS token length (critical for dataset alignment)")
    tok = model.tokenizer
    eos = tok.eos_token
    if not eos:
        record("EOS token length", BLOCKED, "No EOS token found",
               required_change="Find a single-token pad alternative in this model's vocabulary.")
        return

    ids = tok.encode(eos, add_special_tokens=False)
    n = len(ids)
    if n == 1:
        record(
            "EOS token length",
            SAFE,
            f"EOS '{eos}' encodes to exactly 1 token (id={ids[0]}). "
            f"align_cells() arithmetic is valid.",
        )
    else:
        record(
            "EOS token length",
            BLOCKED,
            f"EOS '{eos}' encodes to {n} tokens: {ids}. "
            f"align_cells() assumes exactly 1 token per EOS repetition — "
            f"padding arithmetic will be wrong.",
            required_change=(
                "Find a single-token pad string in this model's vocabulary "
                "(e.g. a punctuation token) and replace EOS padding in "
                "build_dataset.py::align_cells() with that token."
            ),
        )


def check_space_prefix_tokenization(model):
    """Check 4: Does ' ' + answer tokenise with the answer as the first token?

    The pipeline does: token_ids = tokenizer.encode(' ' + gold_answer)[0]
    and uses that ID as the target for logit/probability scoring.
    This is correct only if the first token is the answer token, not a bare space.
    """
    print("\n[CHECK 4] Leading-space tokenisation (critical for all scoring)")
    tok = model.tokenizer

    test_answers = ["Paris", "London", "Berlin", "Washington", "Tokyo"]
    failures = []
    for ans in test_answers:
        spaced_ids  = tok.encode(" " + ans, add_special_tokens=False)
        bare_ids    = tok.encode(ans,       add_special_tokens=False)
        space_ids   = tok.encode(" ",       add_special_tokens=False)

        spaced_str  = tok.decode(spaced_ids[:1]) if spaced_ids else ""
        bare_str    = tok.decode(bare_ids[:1])   if bare_ids   else ""

        # Good: " Paris" → single token whose string contains "Paris"
        # Bad:  " Paris" → [space_token, paris_token] i.e. first token is just " "
        space_is_separate = (spaced_ids and space_ids and
                             spaced_ids[0] in space_ids)
        merged = not space_is_separate and (
            len(spaced_ids) == 1 or ans.lower() in spaced_str.lower()
        )

        if space_is_separate:
            failures.append(
                f"'{ans}': ' {ans}' → {spaced_ids[:3]}... "
                f"(first token is bare space {space_ids}), "
                f"bare ids={bare_ids[:3]}..."
            )

    if not failures:
        record(
            "Leading-space tokenisation",
            SAFE,
            f"All {len(test_answers)} test answers: ' answer' merges space into "
            f"answer token (Ġ-style). get_target_token_id() is correct.",
        )
    else:
        record(
            "Leading-space tokenisation",
            BLOCKED,
            f"{len(failures)}/{len(test_answers)} answers fail: space is a "
            f"SEPARATE token. get_target_token_id() returns the wrong ID.\n"
            + "\n".join(f"       {f}" for f in failures),
            required_change=(
                "Replace the ' ' + gold_answer pattern with a context-aware "
                "token lookup. Encode a full sentence ending in the answer, "
                "extract the last token ID — or use bare encoding without the "
                "leading space if that matches context."
            ),
        )


def check_run_with_cache(model):
    """Check 5: Does run_with_cache work and return expected hook keys?"""
    print("\n[CHECK 5] run_with_cache")
    try:
        prompt = "The capital of France is"
        tokens = model.to_tokens(prompt)
        logits, cache = model.run_with_cache(tokens)
        n_keys = len(cache)
        record("run_with_cache", SAFE,
               f"Returned {n_keys} cache entries. logits shape: {tuple(logits.shape)}")
        return cache
    except Exception as e:
        record("run_with_cache", BLOCKED,
               f"Failed: {type(e).__name__}: {e}",
               required_change="Debug TransformerLens Qwen support.")
        return None


def check_hook_names(model, cache):
    """Check 6: Are standard hook names present in the cache?"""
    print("\n[CHECK 6] Hook names")
    if cache is None:
        record("Hook names", SKIP, "Skipped — cache not available")
        return

    n_layers = model.cfg.n_layers
    expected_hooks = [
        f"blocks.0.hook_resid_post",
        f"blocks.{n_layers-1}.hook_resid_post",
        f"blocks.0.hook_attn_out",
        f"blocks.0.hook_mlp_out",
        f"blocks.0.attn.hook_pattern",
        f"blocks.0.attn.hook_z",
    ]
    missing = [h for h in expected_hooks if h not in cache]
    present = [h for h in expected_hooks if h in cache]

    if not missing:
        record("Standard hook names", SAFE,
               f"All {len(expected_hooks)} expected hooks present in cache.")
    else:
        record("Standard hook names", BLOCKED,
               f"{len(missing)} hooks missing: {missing}. "
               f"Present: {present}",
               required_change=(
                   "Check which hook names TransformerLens uses for this model. "
                   "Run `list(cache.keys())` and inspect."
               ))


def check_residual_stream_shapes(model, cache):
    """Check 7: Residual stream tensor shapes are [batch, seq, d_model]."""
    print("\n[CHECK 7] Residual stream shapes")
    if cache is None:
        record("Residual stream shapes", SKIP, "Skipped — cache not available")
        return

    try:
        n_layers = model.cfg.n_layers
        d_model  = model.cfg.d_model

        resid = cache[f"blocks.0.hook_resid_post"]
        shape = tuple(resid.shape)
        expected_dims = 3
        expected_last = d_model

        if len(shape) == expected_dims and shape[-1] == expected_last:
            record("Residual stream shape", SAFE,
                   f"blocks.0.hook_resid_post shape={shape} "
                   f"(batch, seq, d_model={d_model}) ✓")
        else:
            record("Residual stream shape", BLOCKED,
                   f"Unexpected shape {shape}. Expected (batch, seq, {d_model}).",
                   required_change="Investigate TransformerLens Qwen residual stream layout.")
    except KeyError:
        record("Residual stream shape", BLOCKED,
               "hook_resid_post not in cache.",
               required_change="Check hook names (see Check 6).")
    except Exception as e:
        record("Residual stream shape", ERROR, f"{type(e).__name__}: {e}")


def check_attention_shapes(model, cache):
    """Check 8: Attention hook shapes — critical for GQA models.

    For standard MHA:   hook_pattern shape = (batch, n_heads, seq, seq)
                        hook_z shape       = (batch, seq, n_heads, d_head)
    For GQA models (Qwen2.5-3B has n_kv_heads=8, n_heads=16):
        TransformerLens may expand KV heads to match Q heads, or expose raw KV.
        If raw KV is exposed, head_patching.py will break.
    """
    print("\n[CHECK 8] Attention tensor shapes (GQA check)")
    if cache is None:
        record("Attention shapes", SKIP, "Skipped — cache not available")
        return

    n_heads = model.cfg.n_heads
    try:
        n_kv_heads = model.cfg.n_key_value_heads
    except AttributeError:
        n_kv_heads = n_heads  # standard MHA

    # hook_pattern
    try:
        pattern = cache["blocks.0.attn.hook_pattern"]
        pshape = tuple(pattern.shape)
        # Expected: (batch, n_heads, seq, seq)
        if pshape[1] == n_heads:
            record("hook_pattern shape", SAFE,
                   f"shape={pshape}, head dim={pshape[1]}=n_heads={n_heads}. "
                   f"No GQA expansion issue.")
        elif pshape[1] == n_kv_heads:
            record("hook_pattern shape", BLOCKED,
                   f"shape={pshape}, head dim={pshape[1]}=n_kv_heads={n_kv_heads} "
                   f"(NOT n_heads={n_heads}). Indexing by head in Phase 3b will be wrong.",
                   required_change=(
                       "TransformerLens is NOT expanding KV heads. Phase 3b "
                       "head_patching.py iterates range(n_heads) which will "
                       "index out-of-bounds for KV dimensions."
                   ))
        else:
            record("hook_pattern shape", RISKY,
                   f"Unexpected shape={pshape}. n_heads={n_heads}, n_kv_heads={n_kv_heads}. "
                   f"Needs manual inspection.")
    except KeyError:
        record("hook_pattern shape", BLOCKED,
               "blocks.0.attn.hook_pattern not in cache.",
               required_change="Check hook names.")
    except Exception as e:
        record("hook_pattern shape", ERROR, f"{type(e).__name__}: {e}")

    # hook_z
    try:
        hz = cache["blocks.0.attn.hook_z"]
        zshape = tuple(hz.shape)
        # Expected: (batch, seq, n_heads, d_head)
        if len(zshape) == 4 and zshape[2] == n_heads:
            record("hook_z shape", SAFE,
                   f"shape={zshape}, head dim={zshape[2]}=n_heads={n_heads}. OK.")
        elif len(zshape) == 4 and zshape[2] == n_kv_heads:
            record("hook_z shape", BLOCKED,
                   f"shape={zshape}, head dim={zshape[2]}=n_kv_heads={n_kv_heads}. "
                   f"head_patching will fail.",
                   required_change=(
                       "GQA hook_z not expanded to n_heads. "
                       "Phase 3b head_patching.py assumes (batch, seq, n_heads, d_head)."
                   ))
        else:
            record("hook_z shape", RISKY,
                   f"Unexpected shape={zshape}. Needs manual inspection.")
    except KeyError:
        record("hook_z shape", BLOCKED,
               "blocks.0.attn.hook_z not in cache.",
               required_change="Check hook names.")
    except Exception as e:
        record("hook_z shape", ERROR, f"{type(e).__name__}: {e}")


def check_toy_alignment(model):
    """Check 9: Can we build a minimal toy 5-cell aligned example?

    Creates a tiny 5-cell prompt set and verifies token counts align after
    EOS padding — exactly what build_dataset.py does.  This detects the
    multi-token EOS problem in practice.
    """
    print("\n[CHECK 9] Toy 5-cell token alignment")
    tok = model.tokenizer
    eos = tok.eos_token
    if not eos:
        record("Toy alignment", BLOCKED, "No EOS token — cannot pad.",
               required_change="Find an alternative single-token pad.")
        return

    # Simulate cells A–E (very short, just to exercise the arithmetic)
    base = "Fact 1: The capital of France is Paris.\nFact 2: The mayor of Paris is Anne Hidalgo.\nQ: Who is the mayor?\nAnswer:"
    cells = {
        "A": base,
        "B": "Distractor 1. Distractor 2. Distractor 3. " + base,
        "C": "Step 1: Paris. Step 2: Anne Hidalgo.\n" + base,
        "D": "Distractor 1. Distractor 2. Distractor 3. Step 1: Paris. Step 2: Anne Hidalgo.\n" + base,
        "E": base.replace("Answer:", eos * 15 + "\nAnswer:"),
    }

    counts = {}
    for k, text in cells.items():
        counts[k] = len(tok.encode(text, add_special_tokens=False))

    target = max(counts.values())
    gap_info = {k: target - v for k, v in counts.items()}

    # Try to align by prepending EOS tokens
    aligned_counts = {}
    eos_ids_per_rep = tok.encode(eos, add_special_tokens=False)
    tokens_per_eos = len(eos_ids_per_rep)

    for k, text in cells.items():
        gap = target - counts[k]
        if gap == 0:
            aligned_counts[k] = target
            continue
        if tokens_per_eos == 0:
            aligned_counts[k] = None
            continue
        n_eos = gap // tokens_per_eos
        padded = (eos * n_eos) + text
        aligned_counts[k] = len(tok.encode(padded, add_special_tokens=False))

    all_aligned = all(v == target for v in aligned_counts.values() if v is not None)

    if tokens_per_eos != 1:
        record(
            "Toy 5-cell alignment",
            BLOCKED,
            f"EOS encodes to {tokens_per_eos} tokens. "
            f"Alignment arithmetic (requires 1 token per EOS) is broken. "
            f"Raw gaps: {gap_info}. "
            f"Aligned counts (broken): {aligned_counts}",
            required_change=(
                "Replace EOS padding in build_dataset.py::align_cells() with "
                "a single-token pad character from this model's vocabulary."
            ),
        )
    elif all_aligned:
        record(
            "Toy 5-cell alignment",
            SAFE,
            f"All 5 cells aligned to {target} tokens. "
            f"EOS padding arithmetic works (1 token per EOS). "
            f"Raw counts: {counts}",
        )
    else:
        record(
            "Toy 5-cell alignment",
            RISKY,
            f"EOS is 1 token but alignment still failed. "
            f"Raw counts: {counts}. Aligned: {aligned_counts}. "
            f"Inspect tokenizer boundary effects.",
            required_change=(
                "Review align_cells() convergence loop for this tokenizer."
            ),
        )


def check_deterministic_generation(model):
    """Check 10: Does greedy generation produce a sensible answer?"""
    print("\n[CHECK 10] Deterministic generation")
    try:
        import torch
        prompt = ("Fact 1: The capital of Germany is Berlin.\n"
                  "Fact 2: The mayor of Berlin is Kai Wegner.\n"
                  "Q: Who is the mayor of the capital of Germany?\nA:")
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            out = model.generate(tokens, max_new_tokens=5, temperature=0,
                                 do_sample=False)
        new_toks = out[0, tokens.shape[1]:]
        generated = model.to_string(new_toks)
        contains_answer = "wegner" in generated.strip().lower()
        record(
            "Deterministic generation",
            SAFE if contains_answer else RISKY,
            f"Generated: {generated!r}. "
            f"Answer 'Wegner' {'found' if contains_answer else 'NOT found — model may not handle this format'}.",
        )
    except Exception as e:
        record("Deterministic generation", ERROR,
               f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-3B (or any model) feasibility check for the two-hop pipeline"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B",
                        help="HuggingFace model name to test (default: Qwen/Qwen2.5-3B)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--skip-model-load", action="store_true",
                        help="Skip all model-dependent checks (dry run for static analysis)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"FEASIBILITY CHECK: {args.model}")
    print(f"Device: {args.device}")
    print("=" * 70)

    if args.skip_model_load:
        print("\n[INFO] --skip-model-load set. Skipping all model-dependent checks.")
        record("All model checks", SKIP, "Skipped via --skip-model-load")
    else:
        model = check_model_load(args.model, args.device)
        if model is not None:
            check_tokenizer_basics(model)
            check_eos_token_length(model)
            check_space_prefix_tokenization(model)
            cache = check_run_with_cache(model)
            check_hook_names(model, cache)
            check_residual_stream_shapes(model, cache)
            check_attention_shapes(model, cache)
            check_toy_alignment(model)
            check_deterministic_generation(model)
        else:
            print("\n  Model failed to load. Skipping all further checks.")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("FEASIBILITY SUMMARY")
    print("=" * 70)

    counts = {SAFE: 0, RISKY: 0, BLOCKED: 0, SKIP: 0, ERROR: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        marker = {"SAFE": "✓", "RISKY": "~", "BLOCKED": "✗",
                   "SKIP": "-", "ERROR": "!"}.get(r.status, "?")
        print(f"  [{marker}] {r.status:7s}  {r.name}")
        if r.required_change:
            print(f"             → {r.required_change}")

    print(f"\n  SAFE={counts[SAFE]}  RISKY={counts[RISKY]}  "
          f"BLOCKED={counts[BLOCKED]}  ERROR={counts[ERROR]}  SKIP={counts[SKIP]}")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    if counts[BLOCKED] > 0 or counts[ERROR] > 0:
        print(f"VERDICT: BLOCKED — {counts[BLOCKED]} blocking issue(s), "
              f"{counts[ERROR]} error(s).")
        print("  Full Qwen experiments are NOT runnable today without code changes.")
        print("  Required changes listed above.")
    elif counts[RISKY] > 0:
        print(f"VERDICT: RISKY — {counts[RISKY]} risky assumption(s). "
              "Proceed with caution and verify results manually.")
        print("  Full Qwen experiments MAY be runnable but results need validation.")
    else:
        print("VERDICT: SAFE — all checks passed.")
        print("  Full Qwen experiments appear runnable with existing pipeline.")

    print("=" * 70)

    # Exit non-zero if blocked
    if counts[BLOCKED] > 0 or counts[ERROR] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

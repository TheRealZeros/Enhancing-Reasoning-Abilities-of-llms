import argparse
from transformer_lens import HookedTransformer
import torch
import gc

# ============================================================
# Args
# ============================================================

parser = argparse.ArgumentParser(
    description="Phase 0: Sanity check — load a model and verify basic operation"
)
parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b",
                    help="HuggingFace model name for HookedTransformer")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device: cuda or cpu")
args = parser.parse_args()

# ============================================================
# Model Loading
# ============================================================

print("=" * 60)
print("PHASE 0: Sanity Check")
print("=" * 60)

print(f"\nLoading {args.model}...")

model = HookedTransformer.from_pretrained(
    args.model,
    device=args.device,
    dtype=torch.float16 if args.device == "cuda" else torch.float32,
)

print(f"Model loaded. Device: {model.cfg.device}")
print(f"Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, d_model: {model.cfg.d_model}")

# ============================================================
# Helper: Generate and print only new tokens
# ============================================================

def generate_and_print(model, prompt, label, max_new_tokens=5):
    """Generate from prompt, print only the newly generated tokens."""
    tokens = model.to_tokens(prompt)
    n_prompt_tokens = tokens.shape[1]

    output = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=0,
        do_sample=False
    )

    # Decode only the new tokens (exclude the prompt)
    new_tokens = output[0, n_prompt_tokens:]
    generated_text = model.to_string(new_tokens)

    print(f"\n--- {label} ---")
    print(f"Prompt tokens: {n_prompt_tokens} (includes BOS)")
    print(f"Generated continuation: '{generated_text}'")
    return generated_text

# ============================================================
# Test 1: Direct Few-Shot Prompt
# ============================================================
# Two demonstrations with direct answers, then a test query.
# The model should complete with the answer via pattern matching.

direct_prompt = """Fact 1: The capital of Germany is Berlin.
Fact 2: The mayor of Berlin is Kai Wegner.
Q: Who is the mayor of the capital of Germany?
A: Kai Wegner

Fact 1: The largest ocean is the Pacific Ocean.
Fact 2: The deepest point of the Pacific Ocean is the Mariana Trench.
Q: What is the deepest point of the largest ocean?
A: the Mariana Trench

Fact 1: The capital of France is Paris.
Fact 2: The mayor of Paris is Anne Hidalgo.
Q: Who is the mayor of the capital of France?
A:"""

direct_output = generate_and_print(model, direct_prompt, "DIRECT FEW-SHOT PROMPT")

# ============================================================
# Test 2: Structured Few-Shot Prompt
# ============================================================
# Two demonstrations with explicit Step 1 / Step 2 reasoning,
# then a test query with the same reasoning scaffold.
# The model should still complete with the answer after the steps.

structured_prompt = """Fact 1: The capital of Germany is Berlin.
Fact 2: The mayor of Berlin is Kai Wegner.
Q: Who is the mayor of the capital of Germany?
Step 1: The capital of Germany is Berlin.
Step 2: The mayor of Berlin is Kai Wegner.
A: Kai Wegner

Fact 1: The largest ocean is the Pacific Ocean.
Fact 2: The deepest point of the Pacific Ocean is the Mariana Trench.
Q: What is the deepest point of the largest ocean?
Step 1: The largest ocean is the Pacific Ocean.
Step 2: The deepest point of the Pacific Ocean is the Mariana Trench.
A: the Mariana Trench

Fact 1: The capital of France is Paris.
Fact 2: The mayor of Paris is Anne Hidalgo.
Q: Who is the mayor of the capital of France?
Step 1: The capital of France is Paris.
Step 2: The mayor of Paris is Anne Hidalgo.
A:"""

structured_output = generate_and_print(model, structured_prompt, "STRUCTURED FEW-SHOT PROMPT")

# ============================================================
# Test 3: Activation Caching
# ============================================================

print("\n--- ACTIVATION CACHE TEST ---")

tokens = model.to_tokens(structured_prompt)
logits, cache = model.run_with_cache(tokens)

# Check residual stream at first and last layers
layer0 = cache["resid_pre", 0]
layer_last = cache["resid_post", model.cfg.n_layers - 1]

print(f"Residual stream layer 0 shape:  {layer0.shape}")
print(f"Residual stream layer {model.cfg.n_layers - 1} shape: {layer_last.shape}")
print(f"Cache keys available: {len(cache)} entries")

# Quick check: can we access attention patterns?
attn_pattern = cache["pattern", 0]  # layer 0 attention pattern
print(f"Attention pattern layer 0 shape: {attn_pattern.shape}")
print("Cache test PASSED.")

# Clean up VRAM
del logits, cache, layer0, layer_last, attn_pattern
torch.cuda.empty_cache()
gc.collect()

# ============================================================
# Test 4: Token Count Comparison (alignment preview)
# ============================================================

print("\n--- TOKEN COUNT COMPARISON ---")
direct_tokens = model.to_tokens(direct_prompt)
structured_tokens = model.to_tokens(structured_prompt)

print(f"Direct prompt tokens:     {direct_tokens.shape[1]}")
print(f"Structured prompt tokens: {structured_tokens.shape[1]}")
diff = structured_tokens.shape[1] - direct_tokens.shape[1]
print(f"Difference: {diff} tokens")

if diff != 0:
    print("NOTE: Prompts are NOT token-aligned. This is expected for this")
    print("      sanity check. Phase 1 will enforce strict alignment.")
else:
    print("Prompts are token-aligned.")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("PHASE 0 SUMMARY")
print("=" * 60)

# Simple check on whether the answer is in the output
direct_ok = "hidalgo" in direct_output.strip().lower()
structured_ok = "hidalgo" in structured_output.strip().lower()

print(f"  Model ({args.model}) loaded on {args.device.upper()}: PASS")
print(f"  Direct prompt correct:  {'PASS' if direct_ok else 'FAIL — check output above'}")
print(f"  Structured prompt correct: {'PASS' if structured_ok else 'FAIL — check output above'}")
print(f"  Activation cache works: PASS")
print(f"  Token counts logged:    PASS")
print("=" * 60)

if direct_ok and structured_ok:
    print(f"\nPhase 0 COMPLETE ({args.model}). Ready for Phase 1 (dataset construction).")
else:
    print("\nPhase 0 PARTIAL. Review outputs above.")
    print("If model produces wrong answers, try:")
    print("  1. Simplifying fact domains (use well-known geography)")
    print("  2. Adding a third demonstration")
    print("  3. Using a larger model variant")
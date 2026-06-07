# Final Thesis Results Summary

This summary matches the final thesis submitted in June 2026 and the artifacts
present in this repository. No model inference was rerun for this cleanup.

## Final Scope

The final mechanistic analysis uses two canonical contrasts:

| Model | Contrast | n | Role |
| --- | --- | ---: | --- |
| Pythia-2.8B | A->C, Direct/Clean to Structured/Clean | 38 | Clean reference contrast |
| Qwen2.5-3B | B->D, Direct/Noisy to Structured/Noisy | 104 | Noisy recovery contrast |

Other Qwen contrast files are retained as behavioural context. They are not
presented as canonical Phase 3/4 evidence in the final thesis.

## Behavioural Results

| Model | Cell | Exact match | Containment |
| --- | --- | ---: | ---: |
| Pythia-2.8B | A | 39.5% | 52.0% |
| Pythia-2.8B | B | 42.0% | 50.0% |
| Pythia-2.8B | C | 52.5% | 65.5% |
| Pythia-2.8B | D | 67.0% | 74.5% |
| Pythia-2.8B | E | 0.0% | 0.0% |
| Qwen2.5-3B | A | 85.5% | 90.0% |
| Qwen2.5-3B | B | 18.5% | 27.5% |
| Qwen2.5-3B | C | 65.5% | 79.5% |
| Qwen2.5-3B | D | 66.5% | 98.5% |
| Qwen2.5-3B | E | 0.0% | 0.0% |

Exact match is the primary metric. Containment is a secondary diagnostic for
answers that include the gold answer with additional text.

## Activation Patching

| Model | Contrast | Peak layer | Mean gold-logit delta |
| --- | --- | ---: | ---: |
| Pythia-2.8B | A->C | 31 | +2.5732 |
| Qwen2.5-3B | B->D | 34 | +6.9042 |

Both canonical contrasts show their strongest positive mediation in late
layers. This supports a narrow claim about late-layer answer-state mediation,
not a complete circuit account.

## Component Patching

| Model | Strongest selected component | Mean gold-logit delta |
| --- | --- | ---: |
| Pythia-2.8B | Layer 31 MLP output | +0.3986 |
| Qwen2.5-3B | Layer 35 MLP output | +1.4779 |

Qwen also has a strong selected attention contribution at layer 33
(+1.3440). Component patching covers selected high-effect layers rather than
every component in the network.

## Diagnostics

At the final layer, Pythia gold-answer top-1 rises from 5.3% in Cell A to 97.4%
in Cell C for the selected contrast. Qwen rises from 7.7% in Cell B to 93.3% in
Cell D. Logit lens and attention plots are diagnostic and are not treated as
causal evidence.

## Exploratory Steering

Qwen steering calibration selected layer 32 with the residual-post hook and
alphas from 0.0 to 1.0. At alpha 1.0, the final average-vector run reports a
mean gold-logit increase of 1.6938. Across 160 alpha/example rows, 122 were
helped, 6 were hurt, and 32 were unchanged.

The steering result is interpreted as partial representation-level recovery.
It does not establish that a single average vector fully repairs generation.

## Evidence Locations

- Behaviour: `results/phase_2_behaviour/`
- Containment audit: `results/model_agnostic_evaluation/`
- Canonical contrast data: `dataset/processed/`
- Layer and component patching: `results/phase_3a_layer_patching/` and
  `results/phase_3b_component_patching/`
- Logit lens and attention diagnostics: `results/phase_4a_logit_lens/`,
  `results/phase_4b_attention/`, and `figures/`
- Steering: `results/phase_5a_steering_calibration/`,
  `results/phase_5b_activation_steering/`, and
  `results/phase_5c_steering_analysis/`

## Thesis-Safe Conclusion

Structured prompting has model- and context-dependent effects. In the two
canonical settings where it produces a useful behavioural contrast,
answer-supporting causal effects are concentrated primarily in later
transformer layers.

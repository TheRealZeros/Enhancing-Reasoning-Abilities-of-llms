# Phase 5c Helped vs Hurt Analysis

Source results: `results\phase_5b_activation_steering\qwen2.5-3b\noisy_steering_results.csv`

A row is labelled `helped` when `delta_gold_logit > 0`, `hurt` when `delta_gold_logit < 0`, and `unchanged` when the absolute delta is near zero.

## Alpha Summary

| alpha | helped | hurt | unchanged | mean baseline logit helped | mean baseline logit hurt | mean baseline rank helped | mean baseline rank hurt |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0 | 0 | 32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.25 | 31 | 1 | 0 | 16.5711 | 18.0938 | 10.8065 | 3.0000 |
| 0.5 | 31 | 1 | 0 | 16.5711 | 18.0938 | 10.8065 | 3.0000 |
| 0.75 | 31 | 1 | 0 | 16.5711 | 18.0938 | 10.8065 | 3.0000 |
| 1.0 | 29 | 3 | 0 | 16.5108 | 17.6615 | 11.4138 | 2.3333 |

## Domain Breakdown

| domain | helped | hurt | unchanged |
|---|---:|---:|---:|
| biology | 24 | 0 | 6 |
| culture | 46 | 2 | 12 |
| geography | 44 | 4 | 12 |
| science | 8 | 0 | 2 |

## Top Examples Helped

| example_id | alpha | domain | gold | delta logit | baseline rank | steered rank |
|---|---:|---|---|---:|---:|---:|
| bio_012 | 1.0 | biology | nutrients | 6.2656 | 187 | 13 |
| sci_006 | 1.0 | science | electric charge | 5.6016 | 19 | 1 |
| bio_012 | 0.75 | biology | nutrients | 5.2969 | 187 | 11 |
| bio_046 | 1.0 | biology | nutrients from food | 4.6094 | 25 | 7 |
| bio_038 | 1.0 | biology | oxygen | 4.3750 | 4 | 2 |
| sci_006 | 0.75 | science | electric charge | 4.2891 | 19 | 6 |
| bio_007 | 1.0 | biology | exposure to sunlight | 4.0938 | 20 | 7 |
| bio_046 | 0.75 | biology | nutrients from food | 4.0000 | 25 | 7 |
| bio_038 | 0.75 | biology | oxygen | 3.8281 | 4 | 2 |
| bio_012 | 0.5 | biology | nutrients | 3.7734 | 187 | 16 |

## Top Examples Harmed

| example_id | alpha | domain | gold | delta logit | baseline rank | steered rank |
|---|---:|---|---|---:|---:|---:|
| geo_049 | 1.0 | geography | Africa | -1.2969 | 3 | 5 |
| geo_049 | 0.75 | geography | Africa | -0.7812 | 3 | 5 |
| cul_041 | 1.0 | culture | Southeast Asia | -0.5000 | 1 | 4 |
| cul_029 | 1.0 | culture | Cairo | -0.3438 | 3 | 5 |
| geo_049 | 0.5 | geography | Africa | -0.3281 | 3 | 4 |
| geo_049 | 0.25 | geography | Africa | -0.0312 | 3 | 3 |
| geo_002 | 0.0 | geography | Asia | 0.0000 | 4 | 4 |
| geo_006 | 0.0 | geography | South America | 0.0000 | 1 | 1 |
| geo_007 | 0.0 | geography | Africa | 0.0000 | 2 | 2 |
| geo_019 | 0.0 | geography | Asia | 0.0000 | 1 | 1 |

# Model-Agnostic Answer-Containment Audit

Model: `EleutherAI/pythia-2.8b`

## Purpose

This audit checks whether a generation that fails strict exact-match still contains the gold answer. It is an evaluation-framework expansion over existing behavioural outputs, not a new runnable experiment stage.

Strict exact-match remains the primary behavioural metric used by the experiment. The containment-aware score is a secondary diagnostic for understanding answer quality, especially when structured prompts produce extra wording around an otherwise correct answer.

Labels are mutually exclusive: `exact_match`, `contains_correct_answer`, and `wrong`.

## Per-Cell Summary

| Cell | Total | Exact match | Contains correct answer | Wrong | Strict exact accuracy | Contains-answer accuracy | Wrong rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 200 | 79 | 25 | 96 | 39.5% | 52.0% | 48.0% |
| B | 200 | 84 | 16 | 100 | 42.0% | 50.0% | 50.0% |
| C | 200 | 105 | 26 | 69 | 52.5% | 65.5% | 34.5% |
| D | 200 | 134 | 15 | 51 | 67.0% | 74.5% | 25.5% |
| E | 200 | 0 | 0 | 200 | 0.0% | 0.0% | 100.0% |

## A vs C Interpretation

Cell A strict exact accuracy is 39.5%; Cell C strict exact accuracy is 52.5% (delta +13.0%). Containment-aware accuracy changes from 52.0% to 65.5% (delta +13.5%).

Cell A is Direct/Clean and Cell C is Structured/Clean. A containment-aware comparison can reveal when structured prompting produces the right answer with additional text, even when strict exact-match marks it incorrect.

## B vs D Interpretation

Cell B strict exact accuracy is 42.0%; Cell D strict exact accuracy is 67.0% (delta +25.0%). Containment-aware accuracy changes from 50.0% to 74.5% (delta +24.5%).

Cell B is Direct/Noisy and Cell D is Structured/Noisy. This comparison is useful for checking whether structured reasoning recovers answers in noisy contexts beyond what strict exact-match alone captures.

## Exact-Match Failures That Contain the Gold Answer

| example_id | domain | cell | gold answer | generated answer |
|---|---|---|---|---|
| geo_003 | geography | B | the Australian dollar |  Australian dollar<br><br>Q: What is the capital of Australia?<br>A: |
| geo_003 | geography | C | the Australian dollar |  Australian dollar<br><br>Q: The capital of the United States is Washington, D |
| geo_003 | geography | D | the Australian dollar |  Australian dollar<br><br>Fact 1: The Danube River flows through Vienna.<br> |
| geo_005 | geography | B | the European Union |  Europe<br><br>Q: The most populous country in the European Union is Germany |
| geo_009 | geography | D | largest land area |  the largest land area of any country<br><br>Fact 1: The Danube River |
| geo_016 | geography | A | the United Kingdom |  England<br><br>Fact 3: The capital of the United Kingdom is London.<br> |
| geo_016 | geography | C | the United Kingdom |  United Kingdom<br><br>Q: The capital of the United States is Washington, D |
| geo_018 | geography | D | the Mediterranean Sea |  Mediterranean Sea<br><br>Fact 1: The Danube River flows through Vienna.<br> |
| geo_020 | geography | B | the Atlantic Ocean |  Atlantic Ocean<br><br>Q: The Danube River flows through what country?<br> |
| geo_020 | geography | C | the Atlantic Ocean |  Atlantic Ocean<br><br>Q: The Atlantic Ocean is the largest ocean in the world |
| geo_020 | geography | D | the Atlantic Ocean |  Atlantic Ocean<br><br>Fact 1: The Danube River flows through Vienna.<br> |
| geo_021 | geography | A | the Swedish krona |  Sweden uses the Swedish krona.<br><br>Q: What is the capital |

## Examples Labelled Wrong

| example_id | domain | cell | gold answer | generated answer |
|---|---|---|---|---|
| geo_001 | geography | E | Europe | <br><br>The following is a list of the most common questions and answers to the |
| geo_002 | geography | E | Asia | <br><br>The following is a list of the top 10 most popular questions asked by |
| geo_003 | geography | A | the Australian dollar |  Australia<br><br>Q: What is the capital of Australia?<br>A: Can |
| geo_003 | geography | E | the Australian dollar | <br><br>The following is a list of the top 10 most popular questions asked by |
| geo_004 | geography | E | North America | <br><br>The following is a list of the most common questions and answers to the |
| geo_005 | geography | A | the European Union |  Germany<br><br>Q: The capital of Germany is the most populous country in |
| geo_005 | geography | C | the European Union |  Germany<br><br>Q: The capital of the United States is Washington, D. |
| geo_005 | geography | D | the European Union |  Europe<br><br>Fact 1: The Danube River flows through Vienna.<br>Fact |
| geo_005 | geography | E | the European Union | <br><br>The following is a list of the most common questions and answers to the |
| geo_006 | geography | A | South America |  North America<br><br>Q: The capital of what country?<br>A: United |
| geo_006 | geography | B | South America |  North America<br><br>Q: The first skyscraper was built in what |
| geo_006 | geography | E | South America | <br><br>The following is a list of the top 10 most popular questions asked by |

## Thesis-Safe Interpretation

Use this audit to separate two phenomena: genuinely wrong answers and answers that include the correct answer but fail strict formatting. This is especially relevant for structured prompts, which may encourage explanations or more specific noun phrases.

This diagnostic should be reported alongside strict exact-match accuracy, not instead of it. Exact-match is still the primary behavioural score used to define contrast sets for downstream causal analyses.

## Warning

This is a simple containment audit, not a semantic equivalence judge. It only checks normalised substring containment and may miss paraphrases or count some ambiguous mentions as containing the answer.

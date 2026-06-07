# Model-Agnostic Answer-Containment Audit

Model: `Qwen/Qwen2.5-3B`

## Purpose

This audit checks whether a generation that fails strict exact-match still contains the gold answer. It is an evaluation-framework expansion over existing behavioural outputs, not a new runnable experiment stage.

Strict exact-match remains the primary behavioural metric used by the experiment. The containment-aware score is a secondary diagnostic for understanding answer quality, especially when structured prompts produce extra wording around an otherwise correct answer.

Labels are mutually exclusive: `exact_match`, `contains_correct_answer`, and `wrong`.

## Per-Cell Summary

| Cell | Total | Exact match | Contains correct answer | Wrong | Strict exact accuracy | Contains-answer accuracy | Wrong rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 200 | 171 | 9 | 20 | 85.5% | 90.0% | 10.0% |
| B | 200 | 37 | 18 | 145 | 18.5% | 27.5% | 72.5% |
| C | 200 | 131 | 28 | 41 | 65.5% | 79.5% | 20.5% |
| D | 200 | 133 | 64 | 3 | 66.5% | 98.5% | 1.5% |
| E | 200 | 0 | 0 | 200 | 0.0% | 0.0% | 100.0% |

## A vs C Interpretation

Cell A strict exact accuracy is 85.5%; Cell C strict exact accuracy is 65.5% (delta -20.0%). Containment-aware accuracy changes from 90.0% to 79.5% (delta -10.5%).

Cell A is Direct/Clean and Cell C is Structured/Clean. A containment-aware comparison can reveal when structured prompting produces the right answer with additional text, even when strict exact-match marks it incorrect.

## B vs D Interpretation

Cell B strict exact accuracy is 18.5%; Cell D strict exact accuracy is 66.5% (delta +48.0%). Containment-aware accuracy changes from 27.5% to 98.5% (delta +71.0%).

Cell B is Direct/Noisy and Cell D is Structured/Noisy. This comparison is useful for checking whether structured reasoning recovers answers in noisy contexts beyond what strict exact-match alone captures.

## Exact-Match Failures That Contain the Gold Answer

| example_id | domain | cell | gold answer | generated answer |
|---|---|---|---|---|
| geo_003 | geography | B | the Australian dollar |  the Australian dollar as its currency.<br>Fact 2: The first satellite Sput |
| geo_009 | geography | A | largest land area |  the largest land area of any country<br><br>Fact 1: The Danube River |
| geo_009 | geography | D | largest land area |  Russia has the largest land area of any country.<br>You are an AI assistant that |
| geo_011 | geography | B | Europe |  Madrid is located in Europe.<br>Fact 1: Madrid is located in Europe.<br> |
| geo_015 | geography | B | Asia |  the continent of Asia.<br>Fact 2: The continent of Asia is located on |
| geo_024 | geography | D | the North Atlantic |  the North Atlantic Ocean<br><br>Fact 1: The Earth is tilted on its axis |
| geo_029 | geography | D | the Pacific Ocean |  The Great Barrier Reef is in a country surrounded by the Pacific Ocean.<br><br>Fact  |
| geo_030 | geography | D | the Atlantic Ocean |  The Amazon River flows through a country bordering the Atlantic Ocean.<br><br>Fact 1 |
| geo_032 | geography | D | Russia |  Finland shares a border with Russia.<br>You are an AI assistant that helps people find |
| geo_033 | geography | B | landlocked |  Austria is a landlocked country.<br>Fact 2: The first zoo open to |
| geo_033 | geography | C | landlocked |  a landlocked country.<br><br>Q: Vienna is the capital of a country that is |
| geo_033 | geography | D | landlocked |  Austria is a landlocked country.<br>You are an AI assistant that helps people find |

## Examples Labelled Wrong

| example_id | domain | cell | gold answer | generated answer |
|---|---|---|---|---|
| geo_001 | geography | E | Europe |  |
| geo_002 | geography | B | Asia |  the city of Tokyo is located on what continent?<br>A: the city of Tokyo |
| geo_002 | geography | E | Asia |  |
| geo_003 | geography | E | the Australian dollar |  |
| geo_004 | geography | B | North America |  the continent of Canada.<br>Fact 2: The continent of Canada is located on |
| geo_004 | geography | E | North America |  |
| geo_005 | geography | E | the European Union |  |
| geo_006 | geography | B | South America |  the continent of Brasilia is located in the abdomen.<br><br>Fact 1: Bras |
| geo_006 | geography | E | South America |  |
| geo_007 | geography | B | Africa |  the city of Cairo is located in the capital of what country?<br>A: the |
| geo_007 | geography | E | Africa |  |
| geo_008 | geography | B | Asia |  Beijing is the capital of China.<br>Fact 2: Bats are the only |

## Thesis-Safe Interpretation

Use this audit to separate two phenomena: genuinely wrong answers and answers that include the correct answer but fail strict formatting. This is especially relevant for structured prompts, which may encourage explanations or more specific noun phrases.

This diagnostic should be reported alongside strict exact-match accuracy, not instead of it. Exact-match is still the primary behavioural score used to define contrast sets for downstream causal analyses.

## Warning

This is a simple containment audit, not a semantic equivalence judge. It only checks normalised substring containment and may miss paraphrases or count some ambiguous mentions as containing the answer.

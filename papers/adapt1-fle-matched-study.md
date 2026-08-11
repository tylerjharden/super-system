---
title: "When Strategy Guidance Reduces Errors but Hurts Production: A Matched Adapt-1 × Factorio Study"
authors:
  - Tyler Harden
  - REI Adapt-1 × FLE contributors
date: 2026-08-11
license: mit
tags:
  - reinforcement-learning
  - agents
  - factorio
  - local-llm
  - negative-results
  - reproducible-research
---

# When Strategy Guidance Reduces Errors but Hurts Production

## A matched Adapt-1 × Factorio Learning Environment study with a local code model

**Tyler Harden · REI Adapt-1 × FLE contributors · 11 August 2026**

> **Result in one sentence:** under this short-horizon, local-model protocol,
> trained Adapt-1 guidance did not improve task success or production; it
> reduced execution errors but collapsed every Domain-guided decision to
> `inspect`, eliminating productive behavioral diversity.

## Abstract

We evaluate whether a persistent REI Adapt-1 strategy layer improves the
behavior of a fixed code-synthesis model in the Factorio Learning Environment
(FLE). The experiment uses a fully local Ollama model
(`qwen2.5-coder:7b`, Q4_K_M), a fresh versioned Adapt Domain, domain-scoped
Memory, eight four-step training episodes, and a frozen 40-run comparison
across baseline, Domain+Memory, Domain-only, and Memory-only arms. Evaluation
covers one exposed retention task (`iron_ore_throughput`) and one held-out
transfer task (`iron_gear_wheel_throughput`), with five runs per task and arm.

No arm solved either task (0/10 successes per arm; Wilson 95% CI 0–27.8%).
Baseline achieved the highest aggregate mean final production score (4.3),
compared with 0.0 for warm-frozen and 0.9 for both ablations. On exposed iron
ore, baseline averaged 8.6 while warm-frozen averaged 0.0. Domain-enabled arms
did reduce execution errors (32.5–35.0% versus 52.5% baseline), but all 80
Domain-guided evaluation decisions selected `inspect`. Memory stored seven
recurring-failure records and did not improve held-out transfer. We therefore
find no evidence that Adapt improved FLE outcomes in this protocol. The study
instead identifies a bootstrap lock-in: the first tied decision fell back to
`inspect`, subsequent feedback created support only for that policy, and the
configured learner never explored another declared strategy.

## 1. Research question

The harness was built to answer:

> Does trained Adapt-1 strategy state improve FLE outcomes relative to the same
> model and execution harness without Adapt?

The primary endpoint was task success. Secondary endpoints were final
production score, score area under the trajectory, execution-error rate,
strategy selection, and operational reliability.

This is a negative-results paper. It does not claim Adapt-1 is generally
ineffective; it reports that the tested contract, exposure schedule, and
learner configuration did not improve outcomes.

## 2. System

The causal path was:

```text
FLE observation
  → compact public state
  → Adapt Domain query (+ optional scoped Memory query)
  → high-level strategy and decision_id
  → local model generates FLE Python
  → Factorio executes the exact program
  → public next state and normalized reward
  → decision-bound feedback during training
  → append-only ledger
```

Adapt selected among twelve public strategies:

`inspect`, `gather`, `craft`, `power`, `mine`, `smelt`, `logistics`,
`assemble`, `research`, `debug`, `optimize`, and `verify`.

Adapt did not generate Python or update foundation-model weights. The local
model translated the selected strategy into executable code.

### 2.1 Local model and reproducible setup

The Cloud Agent path requires only an Adapt key; no Anthropic, OpenAI, or
OpenRouter key is required:

```bash
bash .cursor/install.sh
bash .cursor/start.sh
cp .env.example .env
# Set only ADAPT1_API_KEY in .env.
adapt1-fle doctor
```

The setup installs Ollama, pre-pulls the model, starts its OpenAI-compatible
API, and verifies the exact model in `doctor`.

| Property | Value |
|---|---|
| Model | `ollama-qwen2.5-coder:7b` |
| Parameters | 7.6B |
| Format / quantization | GGUF / Q4_K_M |
| Digest | `dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364` |
| Temperature | 0.2 |
| Maximum output | 1,024 tokens |
| Conversation window | system + two recent turns (`max_messages=5`) |

FLE's generated reference prompt was approximately 117 KB and caused local
context truncation and invented imports such as `flapi`. The study therefore
used a compact public FLE API contract. It explicitly states that tools are
already in the persistent REPL, provides valid type signatures and a generic
mining example, rejects non-standard imports, and allows one repair turn for an
invalid model completion. A pre-study pilot improved from production score 0
with the oversized prompt to 3 with the compact prompt.

## 3. Experimental design

### 3.1 Exposure

A fresh Domain, `factorio-local-qwen7b-research-v1`, received:

- four 4-step episodes on `iron_ore_throughput`;
- four 4-step episodes on `iron_plate_throughput`.

This reached Adapt's configured minima of eight sequential episodes and at
least 24 samples.

### 3.2 Frozen comparison

The complete matrix contained 40 runs:

| Arm | Domain | Memory | Measured writes | Runs |
|---|---:|---:|---:|---:|
| `baseline` | no | no | no | 10 |
| `warm_frozen` | trained, frozen | trained, frozen | no | 10 |
| `domain_only` | trained, frozen | no | no | 10 |
| `memory_only` | no | trained, frozen | no | 10 |

Each arm ran five 4-step episodes on:

- exposed retention: `iron_ore_throughput`;
- held-out transfer: `iron_gear_wheel_throughput`.

All arms shared the model digest, prompt contract, temperature, context
window, FLE 0.4.3, Factorio 2.0.73, trajectory length, reward, state
compression, and scoring. The comparison fingerprint was:

`6d9ef73080714dfbb12d5b9346b07ca62c69e96b8fae43e7e47352acbb8ba058`.

The experiment plan and all 40 cells completed with no duplicate or missing
cells, no operational failures, and no ambiguous writes.

### 3.3 Adapt budget

| Phase | Records | Queries |
|---|---:|---:|
| Training Domain feedback | 32 | 32 |
| Training Memory | 7 | 32 |
| Frozen evaluation | 0 | 160 |
| **Study total** | **39** | **224** |

The study stayed well below the available 492-record and 1,987-query
pre-study budget.

### 3.4 Analysis

Pass rates use Wilson 95% intervals. Final scores and error rates are reported
descriptively because five runs per task are underpowered and score
distributions are zero-heavy. We additionally computed exact two-sided
permutation tests on final score as exploratory, unadjusted analyses. These
tests were not pre-registered and are not treated as confirmatory.

## 4. Results

<img src="results/local-research-v1/arm_comparison.svg" alt="Bar charts showing baseline with the highest mean production score and Domain arms with lower execution error rates." />

### 4.1 Primary endpoint: no task successes

Every arm completed 0/10 tasks:

| Arm | Successes | Pass rate (Wilson 95% CI) |
|---|---:|---:|
| Baseline | 0/10 | 0.0% (0.0–27.8%) |
| Warm-frozen | 0/10 | 0.0% (0.0–27.8%) |
| Domain-only | 0/10 | 0.0% (0.0–27.8%) |
| Memory-only | 0/10 | 0.0% (0.0–27.8%) |

The primary endpoint provides no evidence of benefit for any Adapt arm.

### 4.2 Secondary endpoints

| Arm | N | Mean final score | Mean score AUC | Execution-error rate | Adapt selections |
|---|---:|---:|---:|---:|---:|
| Baseline | 10 | **4.3** | **9.55** | 52.5% | 0% |
| Warm-frozen | 10 | 0.0 | 0.00 | **32.5%** | 100% |
| Domain-only | 10 | 0.9 | 1.35 | 35.0% | 100% |
| Memory-only | 10 | 0.9 | 1.95 | 55.0% | 0% |

Domain guidance made generated programs less error-prone but not more
productive. Warm-frozen reduced errors by 20 percentage points relative to
baseline while reducing mean final score by 4.3.

### 4.3 Exposed retention task

Final scores on `iron_ore_throughput` were:

| Arm | Five run scores | Mean | Execution-error rate |
|---|---|---:|---:|
| Baseline | 3, 3, 0, 31, 6 | **8.6** | 55% |
| Warm-frozen | 0, 0, 0, 0, 0 | 0.0 | **25%** |
| Domain-only | 6, 0, 3, 0, 0 | 1.8 | **20%** |
| Memory-only | 0, 0, 6, 0, 3 | 1.8 | 55% |

The exploratory exact permutation comparison of baseline versus warm-frozen
gave an unadjusted two-sided \(p=0.0476\). This does **not** establish a
general harmful effect: it is post hoc, based on five runs per arm, influenced
by one baseline score of 31, and not adjusted for multiple comparisons. It
does show that the tested Adapt state did not retain the baseline's productive
behavior.

### 4.4 Held-out transfer task

Every `iron_gear_wheel_throughput` run scored 0. Domain guidance reduced the
error rate from 50% baseline to 40% warm-frozen, but this did not produce any
gears. No arm demonstrated transfer.

### 4.5 Policy collapse

The most informative result is the strategy trace:

| Arm | `inspect` | `debug` | Other policies |
|---|---:|---:|---:|
| Baseline controller | 13 | 27 | 0 |
| Warm-frozen Domain | **40** | 0 | 0 |
| Domain-only | **40** | 0 | 0 |
| Memory-only controller | 11 | 29 | 0 |

Training began with a tie across all twelve hypotheses. The application
fallback selected `inspect`; feedback then created the only observed policy
score for `inspect`. All subsequent training and frozen Domain selections were
also `inspect`, even as its raw learned score fell with repeated failures.
Thus the Domain replaced the baseline controller's error-responsive `debug`
strategy with a single conservative action label.

### 4.6 Failure-only Memory

Training wrote seven Memory records. Every admission reason was
`recurring_failure`; none represented success or meaningful progress.
Memory was non-empty on the exposed iron-ore task and empty on held-out gears
because evidence was correctly scoped by Domain and task. Memory-only matched
Domain-only mean production (1.8) but did not reduce execution errors. Adding
failure-only Memory to the Domain produced the lowest production
(warm-frozen, 0.0).

## 5. Interpretation

The experiment answers the stated question for this protocol: **Adapt did not
improve FLE outcomes.**

The result is more specific than “the model was too weak.” The local model
produced valid FLE programs and baseline iron-ore production, including a run
scoring 31. Adapt changed the information condition in a measurable way: it
reduced execution errors and eliminated strategy diversity. The failure was
that safer programs did not advance the objective.

The ablations suggest two mechanisms:

1. **Domain bootstrap lock-in.** An initial tie deterministically chose
   `inspect`; feedback existed only for the chosen policy, so untried policies
   never entered the learned score set.
2. **Negative-only Memory.** Selective admission stored recurring failures but
   no successful lesson. This context did not help transfer and may have made
   warm behavior more conservative.

## 6. Limitations

- Five runs per task and arm yield wide uncertainty.
- No arm solved a task, limiting conclusions about success.
- Arm order was fixed (baseline → warm → Domain → Memory), not randomized.
- Ollama sampling seeds were not fixed or recorded, so episode indices are not
  true paired replicates.
- The four-action horizon strongly favors bootstrap behavior and may be too
  short for gear automation.
- The exposed-task baseline distribution was skewed by one score-31 run.
- Only one local model, one Domain contract, and two evaluation tasks were
  tested.
- The exploratory permutation result was post hoc and unadjusted.

These limitations prevent a broad claim about Adapt-1. They do not change the
descriptive finding that this exact trained state underperformed its matched
baseline.

## 7. Recommended follow-up

A valid next study should change the learning protocol before adding more
episodes:

1. Force coverage of all declared strategies during exposure, or configure a
   documented exploratory policy instead of exploit-only bootstrap.
2. Do not admit recurring failures to long-term Memory until at least one
   positive exemplar exists; separately ablate positive-only and
   failure-diagnostic Memory.
3. Randomize or interleave arm order and record model sampling seeds.
4. Extend trajectories enough to complete smelting and assembly.
5. Pre-register primary and secondary tests and increase per-cell replication.
6. Preserve the exact model digest and comparison fingerprint as done here.

Repeating the current configuration would mostly add evidence about the
already-observed `inspect` lock-in rather than test broader strategy learning.

## 8. Reproducibility and data

Repository branch: `cursor/adapt-api-domain-contract-1034`  
Harness commit used by the experiment: `31294b1`  
Experiment ID: `experiment-20260811T041432Z-3ff8ac0a`

Commands:

```bash
adapt1-fle --config configs/local-research.yaml train \
  --curriculum configs/curriculum.local-research.v1.yaml \
  --steps 4

adapt1-fle --config configs/local-research.yaml evaluate \
  --curriculum configs/curriculum.local-research.v1.yaml \
  --arm baseline \
  --arm warm_frozen \
  --arm domain_only \
  --arm memory_only \
  --task iron_ore_throughput \
  --task iron_gear_wheel_throughput \
  --episodes 5 \
  --steps 4 \
  --continue-on-error

adapt1-fle --config configs/local-research.yaml report \
  --experiment-id experiment-20260811T041432Z-3ff8ac0a \
  --output reports/local-research-v1
```

Machine-readable summary:
[`summary.json`](results/local-research-v1/summary.json)  
Arm metrics:
[`arm_metrics.csv`](results/local-research-v1/arm_metrics.csv)  
Complete generated benchmark:
[`benchmark.json`](results/local-research-v1/benchmark.json)

Secrets are excluded. Raw local ledgers remain gitignored because they contain
large prompts, model outputs, and execution traces.

## 9. Conclusion

This matched research preview produced a clear negative result. Adapt-1
guidance reduced FLE execution errors but did not improve success, production,
or held-out transfer. The trained Domain collapsed to `inspect`, and scoped
Memory contained only recurring failures. Baseline remained more productive
on the exposed task.

The harness now provides what the earlier operational report lacked: a complete
comparison matrix, immutable protocol fingerprint, explicit quota accounting,
ablation results, and a falsifiable diagnosis. The next experiment should test
exploration and positive-evidence admission—not simply repeat more episodes of
the same locked policy.

## References

1. Jack Hopkins et al. *Factorio Learning Environment*.  
   https://github.com/JackHopkins/factorio-learning-environment
2. Ollama. *Run large language models locally*.  
   https://ollama.com/
3. Qwen Team. *Qwen2.5-Coder*.  
   https://huggingface.co/Qwen/Qwen2.5-Coder-7B

---
title: "REI Adapt-1 Online Strategy Learning in the Factorio Learning Environment: Research Preview"
authors: Tyler J. Harden, REI Adapt-1 x FLE contributors
affiliations: Independent research preview
date: 2026-08-11
type: ml-experiment-report
tags:
  - machine-learning
  - reinforcement-learning
  - agents
  - factorio
  - adapt-1
  - fle
  - sequential-decision-making
status: research-preview
---

# REI Adapt-1 Online Strategy Learning in the Factorio Learning Environment: Research Preview

**Machine Learning Experiment Report**

**Researchers**: Tyler J. Harden, REI Adapt-1 x FLE contributors  
**Date**: 2026-08-11  
**Status**: Research preview (incomplete benchmark; operational validation complete)

---

## Executive Summary

We built and exercised a research-preview harness that trains a persistent REI Adapt-1 strategy substrate inside the Factorio Learning Environment (FLE 0.4.3). Adapt-1 selects among twelve public factory strategies from compact public state; an FLE-compatible model translates that structured selection into executable Python; exact FLE transitions supply sequential feedback back to the same Adapt decision.

This report documents a successful end-to-end operational pass:

1. `adapt1-fle doctor` reported authenticated Adapt readiness, local Factorio RCON, FLE registry, and model credential presence.
2. Domain create initially failed against NeuroAdapt schema v5; after aligning the Domain contract, `factorio-strategy-v2` was created (HTTP 201).
3. The authenticated Adapt live smoke (`tests/test_live.py`) passed.
4. A short cold-online train episode completed three model-backed steps on `iron_ore_throughput` (run `iron-ore-throughput-20260811T025031Z-0fae0832`) using local Ollama `qwen2.5-coder:3b` after Anthropic API credit exhaustion blocked the default Claude model.

The episode did **not** solve the throughput task. It is evidence that the Adapt↔FLE closed loop runs with decision-bound feedback, not that Adapt improves FLE performance. Matched-condition baseline vs warm-frozen evaluation remains future work.

### Key Findings

- The live NeuroAdapt Domain create API rejects application-only fields (`schema.actions`, `schema.outcomes`, `schema.result_contract`) and requires hypotheses with `name` plus list-valued `predicts`, and `query_templates.feedback_outcomes`.
- Authenticated Domain create + query + feedback succeeded for `factorio-strategy-v2`.
- Cold-online training completed 3/3 steps with Adapt selections, model-generated Python, FLE execution, and accepted feedback exchanges.
- Provider fragility matters: Anthropic returned `credit balance is too low`; a local Ollama coder preserved the model-backed path but produced mostly invalid FLE actions.

### Recommendations

- Keep Domain IDs versioned whenever the create contract drifts (`factorio-strategy-v2`).
- Treat provider outages as first-class: document Ollama/OpenRouter fallbacks and repair-turn code extraction.
- Do not claim Adapt superiority until repeated held-out arms with a capable code model are complete.

---

## 1. Objective

### 1.1 Research Question

Can a persistent Adapt-1 Domain learn useful high-level factory strategy support from exact FLE execution outcomes, when code synthesis remains an external model responsibility?

### 1.2 Success Criteria

| Criterion | Target | This preview |
|---|---|---|
| Doctor preflight | `ok: true` with auth | Met |
| Domain create/status | Versioned Domain exists | Met (`factorio-strategy-v2`) |
| Authenticated Adapt smoke | Live create/select/delete | Met |
| Model-backed train episode | ≥1 Adapt-enabled trajectory with feedback | Met (3 steps) |
| Task success / matched benchmark | Pass-rate advantage vs baseline | Not attempted |

### 1.3 Constraints

- Single writer for Adapt Domain online state.
- No foundation-model weight updates; only Adapt Domain/Memory state.
- Text/structured observations only (no sprites/vision).
- Anthropic credits unavailable during the model-backed episode.

---

## 2. System Under Test

### 2.1 Architecture

```text
FLE observation
  -> compact public state
  -> Adapt-1 Domain query (+ Memory query)
  -> structured strategy, support, decision_id
  -> FLE model generates validated Python
  -> FactorioGymEnv.step(Action)
  -> exact transition, scores, error, terminal state
  -> bounded sequential feedback + selective Memory write
  -> append-only audit ledger
```

### 2.2 Public strategy vocabulary

`inspect`, `gather`, `craft`, `power`, `mine`, `smelt`, `logistics`, `assemble`, `research`, `debug`, `optimize`, `verify`

Policies are declared as named hypotheses under relation `advances_goal`. Feedback outcomes are `progress`, `no_progress`, `execution_error`, and `success`.

### 2.3 Software versions

| Component | Version / ID |
|---|---|
| Harness | adapt1-fle 0.1.0 |
| FLE | 0.4.3 |
| Factorio image | factoriotools/factorio:2.0.73 |
| Adapt API | v0.1.0 / schema_version 5 |
| Domain | factorio-strategy-v2 (revision 2) |
| Code model (planned) | claude-sonnet-4-20250514 |
| Code model (used) | ollama-qwen2.5-coder:3b |
| Git SHA (train run) | 013781f007c1fde92bdf4cb2b749fdae3c9b9669 |

---

## 3. Methods

### 3.1 Domain contract alignment

The previous local Domain YAML embedded actions/outcomes/result contracts inside `schema` and used hypothesis `id` with string `predicts`. The live create endpoint returned HTTP 422 for those fields. Revision 2 maps policies into hypotheses (`name`, `policy`, list `predicts`) and outcomes into `query_templates.feedback_outcomes`, keeping schema limited to entities, relations, event types, and signals.

### 3.2 Training protocol (short episode)

```bash
adapt1-fle doctor
adapt1-fle domain create
RUN_LIVE_TESTS=1 pytest -m live tests/test_live.py
ADAPT1_FLE_MODEL=ollama-qwen2.5-coder:3b OLLAMA_API_KEY=ollama \
  adapt1-fle run \
    --mode train \
    --env-id iron_ore_throughput \
    --steps 3 \
    --benchmark-arm cold_online
```

Cold-online runs mint a fresh Domain ID suffix so one trial cannot leak state into another while still exercising writes.

### 3.3 Reliability controls

- Mutating Adapt calls are never blindly retried.
- Append-only JSONL ledger records decision/action preparation, interactions, and completion.
- Policy generation now performs one repair turn and falls back to the last fenced Python block before rejecting a completion.

---

## 4. Results

### 4.1 Preflight (`adapt1-fle doctor`)

| Check | Result |
|---|---|
| Python | 3.12.13 ok |
| FLE | 0.4.3, 30 environments, `iron_ore_throughput` present |
| Factorio RCON | 127.0.0.1:27000 ok |
| Adapt health/version | ok / v0.1.0 schema 5 |
| Adapt authentication | ok |
| Model credential | present (Anthropic initially; Ollama for train) |
| Ledger path | `.fle/adapt1/runs` writable |

### 4.2 Domain create

- First attempt against the old contract: HTTP 422 validation errors on `schema.actions`, `schema.outcomes`, `schema.result_contract`, hypothesis `id`/`predicts`, and custom `query_templates.strategy_selection`.
- After contract alignment: HTTP 201 create for `factorio-strategy-v2`; `domain status` returned the persisted hypotheses and feedback outcomes.

### 4.3 Authenticated Adapt smoke

`tests/test_live.py::test_live_adapt_health_and_authenticated_domain` **PASSED** (create ephemeral Domain, select a policy, delete Domain).

### 4.4 Short model-backed train episode

| Field | Value |
|---|---|
| Run ID | `iron-ore-throughput-20260811T025031Z-0fae0832` |
| Arm / mode | `cold_online` / `train` |
| Steps completed | 3 |
| Completion status | `trajectory_limit` |
| Task success | false |
| Final production score | 0.0 |
| Final automated score | -30.0 |

Step trace:

| Step | Adapt policy | Selection source | Execution error | Normalized reward |
|---:|---|---|---|---:|
| 0 | inspect | adapt_1 | true | -0.25 |
| 1 | inspect | adapt_1 | true | -0.25 |
| 2 | inspect | adapt_1 | false | 0.0 |

Each interaction recorded an accepted Adapt `feedback_exchange`. Memory writes were not triggered (no success / high reward / recurring error category admission). The local coder often emitted non-FLE Python (for example importing unrelated libraries), which is a model-quality failure, not an Adapt transport failure.

### 4.5 What this does *not* show

- No matched baseline comparison.
- No warm-frozen held-out evaluation.
- No claim that Adapt improves throughput, pass rate, or sample efficiency.
- Small-N operational evidence only.

---

## 5. Discussion

The preview separates three failure modes that are easy to conflate:

1. **Contract drift** — Domain create 422s until the payload matches schema v5.
2. **Provider availability** — Anthropic credit exhaustion blocked the intended Claude synthesizer.
3. **Code-model competence** — a 3B local coder completed the loop but did not produce useful FLE actions.

Adapt selection and feedback pathing worked under cold-online conditions. The next scientific step is unchanged from the harness design: repeated held-out arms (`baseline`, `warm_frozen`, `domain_only`, `memory_only`) with a capable synthesizer and Wilson intervals on pass rate.

---

## 6. Limitations

- Single short episode; no statistical comparison.
- Local Ollama model is not the production synthesizer target.
- Vision/sprites disabled.
- Shared-Domain parallel writers intentionally disabled.
- Adapt support/confidence is not treated as calibrated correctness.

---

## 7. Reproducibility

Repository: `github.com/tylerjharden/super-system`  
Branch containing Domain contract + parser hardening: `cursor/adapt-api-domain-contract-1034`  
Primary configs: `configs/domain.factorio.v1.yaml`, `configs/research-preview.yaml`, `configs/curriculum.v1.yaml`  
Ledger root (local, gitignored): `.fle/adapt1/runs/`

Commands used for this report are listed in §3.2. Secrets are never written into manifests; credential-bearing fields are redacted.

---

## 8. Conclusion

The Adapt-1 × FLE research-preview harness can create a versioned Domain against the live NeuroAdapt API, authenticate, select strategies, synthesize model-backed actions, execute them in Factorio, and bind sequential feedback into an append-only ledger. A three-step cold-online episode completed under local model fallback. Establishing whether Adapt improves FLE outcomes requires the planned matched-condition evaluation suite and a capable code-synthesis provider.

---

## References

1. Hopkins et al., Factorio Learning Environment — https://github.com/JackHopkins/factorio-learning-environment  
2. REI NeuroAdapt API (`rei-neuroadapt-api.reilabs.org`), schema version 5  
3. Project docs: `README.md`, `docs/DOMAIN_DESIGN.md`, `docs/EXPERIMENTS.md`, `docs/OPERATIONS.md`

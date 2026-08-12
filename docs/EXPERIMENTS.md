# Experiment methodology

## Success state

The preview is successful when a matched-condition benchmark can show whether
Adapt-1 improves FLE behavior relative to the same model and execution harness
without Adapt. A valid result requires:

- identical FLE version, task definition, trajectory length, model, prompt/tool
  documentation, and score calculation;
- explicit Adapt exposure state;
- repeated independent episodes;
- frozen reads during held-out scoring;
- exact records of fallback, overrides, generated code, and transport failures;
- no hidden solver labels in the learner view.

## Curriculum

`configs/curriculum.v1.yaml` orders exposure by dependency:

1. extraction;
2. material/fluid processing;
3. mechanics and logistics;
4. circuits and advanced intermediates;
5. science;
6. held-out advanced products;
7. open-play transfer.

Each curriculum job has a stable ID. Training resumes by finding a confirmed
completion event for that job. Failed or ambiguous jobs block automatic rerun
until an operator reconciles Adapt and FLE state.

## Comparison arms

| Arm | Purpose |
|---|---|
| `baseline` | Same model/harness with no Adapt gateway |
| `cold_online` | Fresh Domain learning during the measured episode |
| `warm_frozen` | Previously exposed Domain and Memory, no measured writes |
| `domain_only` | Frozen learned Domain without Memory |
| `memory_only` | Frozen Memory with deterministic strategy controller |

For cold-online evaluation, each run receives a fresh Domain ID so one trial
cannot leak learned state to another. Warm-frozen trials intentionally share the
trained Domain but make non-mutating calls.

## Metrics

Primary:

- pass@N / task success;
- throughput proportion and final production score;
- automated production score;
- score area under the trajectory and growth;
- steps to success.

Diagnostics:

- Python/FLE execution error rate;
- Adapt selection, fallback, and abstention rates;
- model tokens and latency;
- Adapt latency;
- exact evidence IDs and decision IDs;
- failures separated by model, Adapt transport, Adapt contract, FLE execution,
  and task outcome.

Pass-rate reports use a Wilson 95% interval. Small samples must not be presented
as conclusive.

## Local-model research preview

`configs/curriculum.local-research.v1.yaml` is the reproducible,
resource-bounded protocol for Cloud Agents without a hosted-model key. It uses
Ollama `qwen2.5-coder:7b`, eight four-step training episodes, one exposed
retention task plus one held-out transfer task, five evaluation episodes per
task, and four frozen comparison arms:

- `baseline`: deterministic public-phase controller, no Adapt calls;
- `warm_frozen`: trained Domain plus domain-scoped Memory;
- `domain_only`: trained Domain without Memory;
- `memory_only`: deterministic controller plus domain-scoped Memory.

At four steps, the protocol plans at most 64 records and 224 queries. Exact
Ollama digest/quantization metadata is stored in each run manifest and included
in the comparison fingerprint. The local API prompt is compact enough to fit
the model context and uses only public FLE tool semantics.

## Run sequence

```bash
# Baseline smoke
adapt1-fle run --mode baseline --env-id iron_ore_throughput --steps 64

# Create/verify the versioned Domain
adapt1-fle domain create

# Curriculum exposure
adapt1-fle train

# Repeated held-out comparison
adapt1-fle evaluate \
  --arm baseline \
  --arm warm_frozen \
  --arm domain_only \
  --arm memory_only \
  --episodes 8

# Aggregate
adapt1-fle report --output reports/experiment-001
```

## Follow-up protocol

`configs/experiment.followup.v1.yaml` preregisters the longer-horizon study
before any new Domain write. Its changes are deliberate:

- UCB is declared in the Domain contract, while exposure also forces balanced
  shuffled coverage so every policy receives feedback once per 12-step episode;
- positive-only and failure-diagnostic Memory are materialized into separate
  metadata profiles, with failure admission gated on prior same-task positive
  evidence;
- Memory queries use Domain scope to permit held-out transfer;
- evaluation cells are shuffled with seed `20260812`;
- each task/episode shares a model seed across arms;
- six episodes per task/arm exceed the first study's five;
- primary and secondary tests are defined before execution.

The expected maximum is 129 records (including Domain creation) and 816
queries. The exact preregistration hash is stored in the experiment plan and
every run manifest.

One `evaluate` invocation prints and stores an immutable experiment ID. Reports
ignore curriculum/non-benchmark runs, require one experiment ID, and reject
mixed comparison fingerprints. Use `--experiment-id` when a ledger root
contains multiple evaluation invocations.

## Information-condition checks

Every manifest records:

- harness git SHA and version;
- FLE version;
- model identifier;
- task and trajectory length;
- Domain ID, revision, and contract hash;
- benchmark arm and held-out flag;
- redacted settings.

Every interaction records:

- compact decision-time state;
- complete normalized Adapt result and raw response;
- selected-by provenance;
- Memory result;
- exact Python and model usage;
- exact model input messages and staged decision/action WAL records;
- FLE consequence;
- raw and normalized reward;
- feedback and Memory-write response;
- ambiguity/failure classification.

## Frozen evaluation

Frozen mode:

- requires an existing Domain;
- sends `allow_exploration: false`;
- sends `update_memory_state: false`;
- does not call Domain feedback;
- does not store Memory;
- preserves returned learner state/version fields in raw responses.

If an endpoint is observed changing learner state despite this contract,
evaluation must stop and the result must be excluded.

## Claim policy

Do not claim "best", "state of the art", or superiority from:

- a single task or episode;
- unmatched FLE/model versions;
- online learning measured against a frozen baseline without disclosure;
- a changed policy score without evidence that the application executed it;
- confidence/support values treated as correctness probabilities;
- runs with silent fallback or unresolved ambiguous writes.

A leaderboard claim requires enough repeated public-task evidence for external
review and reproduction.

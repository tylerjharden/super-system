# REI Adapt-1 x Factorio Learning Environment

Research-preview harness for training a persistent REI Adapt-1 strategy substrate
inside the [Factorio Learning Environment (FLE)](https://github.com/JackHopkins/factorio-learning-environment).

The system learns which **high-level factory strategy** is useful in each public
Factorio state, asks an FLE-compatible model to translate that structured result
into Python, executes the exact program, and binds the observable consequence
back to the same Adapt-1 decision.

## What this trains

Adapt-1 is the continuing reasoning and learning substrate. It receives compact
structured observations, chooses among declared public policies, and updates
from decision-bound sequential feedback. It does not generate FLE Python.
Code synthesis remains an application-boundary responsibility handled through
FLE's model gateway.

This project trains online Adapt-1 state and cross-episode evidence. It does not
change foundation-model weights.

```text
FLE observation
  -> compact public state
  -> Adapt-1 Domain query + Memory query
  -> structured strategy, support, decision_id
  -> FLE model generates validated Python
  -> FactorioGymEnv.step(Action)
  -> exact transition, scores, error, terminal state
  -> bounded sequential feedback + selective Memory write
  -> append-only audit ledger
```

## Architecture

| Component | Responsibility |
|---|---|
| FLE 0.4.3 | Factorio task registry, observations, Python REPL, execution, verification, scoring |
| Adapt-1 Domain | Persistent strategy support and sequential credit across changing states |
| Adapt-1 Memory | Selective reusable evidence from successes, meaningful gains, and recurring failures |
| FLE model gateway | Translate current state and Adapt strategy into executable Python |
| Research ledger | Preserve input, result, commitment, code, consequence, feedback, versions, and failures |

### State boundaries

- **Operational record:** complete before/after observations, exact code and
  output, raw scores, normalized reward, requests/responses, timing, and IDs.
- **Learner view:** public state, available strategy, executed strategy,
  observable next state, reward, and episode boundary.
- **Application-only context:** benchmark arm, held-out labels, fallback
  provenance, and report statistics. These are never answer-bearing learner
  inputs.

## Why ActiveGraph is not included

ActiveGraph's event-sourced replay and fork model is promising, but adding it
now would create a third state authority beside FLE checkpoints and Adapt
learner state. FLE already provides trajectory/game-state primitives, and this
repository's narrow JSONL ledger directly records the causal chain Adapt's
documentation requires. Adapt decisions also receive new sealed decision IDs
when replayed, so graph forks cannot naively replay feedback.

The initial preview therefore uses FLE + Adapt-1 only. ActiveGraph remains an
extension option when graph-reactive multi-behavior orchestration—not Factorio
learning itself—becomes the research target.

## Prerequisites

- Linux with Docker
- Python 3.12
- Factorio server image `factoriotools/factorio:2.0.73`
- FLE `>=0.4.3,<0.5`
- REI Adapt-1 bearer key
- Ollama with `qwen2.5-coder:7b` (installed automatically), or a hosted-model key

The Cursor environment installs these automatically:

```bash
bash .cursor/install.sh
bash .cursor/start.sh
```

For a normal checkout:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]'
fle cluster start -n 1
```

## Configuration

```bash
cp .env.example .env
```

Set the Adapt key. Local code synthesis is the default and needs no third-party
model API key:

```dotenv
ADAPT1_API_KEY=<rei-unit-api-key>
ADAPT1_FLE_MODEL=ollama-qwen2.5-coder:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
```

`REI_API_KEY`, `REI_SECRET_TOKEN`, and `REI_UNIT_API_KEY` are accepted as
compatibility aliases when `ADAPT1_API_KEY` is absent.

Run diagnostics:

```bash
adapt1-fle doctor
```

`doctor` checks Python, FLE, the task registry, local Factorio RCON, public
Adapt health/version, authenticated Domain access when a key is present, the
Ollama runtime and exact model digest (or hosted credential), and the ledger
path. It never prints secret values.

### Local Ollama setup

The Cloud Agent scripts install Ollama, pre-pull `qwen2.5-coder:7b`, and start
its OpenAI-compatible API on each boot:

```bash
bash .cursor/install.sh
bash .cursor/start.sh
curl -fsS http://127.0.0.1:11434/api/tags | jq .
```

For an existing Linux checkout:

```bash
sudo apt-get install -y zstd
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
# In another shell:
ollama pull qwen2.5-coder:7b
```

The harness sends local models a compact FLE API contract rather than FLE's
roughly 117 KB generated reference prompt. FLE tools are already present in the
game REPL and must be called directly; generated imports of non-standard
modules such as `flapi` are rejected and repaired before execution.

## Quickstart

### 1. Create or verify the versioned Domain

```bash
adapt1-fle domain create
adapt1-fle domain status
```

Domain schema changes require a new `domain_id`; the client rejects accidental
contract drift instead of mixing incompatible evidence.

### 2. Run a baseline

```bash
adapt1-fle run \
  --mode baseline \
  --env-id iron_ore_throughput \
  --steps 64 \
  --benchmark-arm baseline
```

### 3. Run a short online Adapt episode

```bash
adapt1-fle run \
  --mode train \
  --env-id iron_ore_throughput \
  --steps 64 \
  --benchmark-arm cold_online
```

### 4. Train the curriculum

```bash
adapt1-fle train --curriculum configs/curriculum.v1.yaml
```

Training is serial by design: Adapt's ordered online state has exactly one
writer. The command resumes by skipping curriculum jobs with a confirmed
completion event. A failed or ambiguous prior job blocks automatic rerun until
an operator reconciles server state.

### 5. Run frozen held-out evaluation

```bash
adapt1-fle evaluate \
  --arm baseline \
  --arm warm_frozen \
  --episodes 8
```

Frozen evaluation sets both `allow_exploration: false` and
`update_memory_state: false`, sends no Domain feedback, and performs no Memory
writes.

### Resource-bounded local research protocol

For Adapt accounts with a 500-record / 2,000-query cap, the bundled local study
uses a fresh versioned Domain and domain-scoped Memory:

```bash
adapt1-fle train \
  --curriculum configs/curriculum.local-research.v1.yaml \
  --steps 6

adapt1-fle evaluate \
  --curriculum configs/curriculum.local-research.v1.yaml \
  --arm baseline \
  --arm warm_frozen \
  --arm domain_only \
  --arm memory_only \
  --episodes 3 \
  --steps 6 \
  --continue-on-error
```

The planned maximum is 48 records (24 Domain feedback + at most 24 selective
Memory writes) and 192 queries (48 during training and 144 during frozen
evaluation). Baseline makes no Adapt calls. The two held-out tasks and all four
arms use the same model, prompt, FLE version, trajectory length, and scoring
contract. Three episodes per task are a research preview, not high-powered
evidence; reports retain Wilson intervals and operational failures.

### 6. Generate a report

```bash
adapt1-fle report --output reports/latest
```

If the ledger root contains more than one evaluation invocation, select the
printed experiment ID:

```bash
adapt1-fle report --experiment-id <experiment-id> --output reports/latest
```

This writes:

- `benchmark.json`: complete machine-readable aggregate;
- `benchmark.md`: pass rate with Wilson 95% interval, score, automation,
  execution-error, Adapt-selection, fallback, and abstention metrics.

## Benchmark arms

| Arm | Domain | Memory | Writes |
|---|---:|---:|---:|
| `baseline` | no | no | no |
| `cold_online` | fresh | no | yes |
| `warm_frozen` | trained | yes | no |
| `domain_only` | trained | no | no |
| `memory_only` | no | yes | no |

The baseline follows the same observation, prompt, Python validation, FLE
execution, ledger, and scoring path. Only Adapt participation differs.

## Adapt strategy contract

The Domain declares twelve public policies:

`inspect`, `gather`, `craft`, `power`, `mine`, `smelt`, `logistics`,
`assemble`, `research`, `debug`, `optimize`, and `verify`.

Factorio is modeled as a real sequential task because actions change later
states and delayed/terminal outcomes should revise earlier choices. Feedback
contains:

- stable run, episode, trial, interaction, and event IDs;
- ordered numeric step;
- exact relation and executed policy;
- sealed `decision_id` when Adapt returns one;
- public next-state object;
- immediate and sequential reward;
- terminal flag.

No interval-policy route is used: Factorio requires categorical strategy plus
external code execution, not interval retention. No custom transition learner
is enabled in v1; the simpler documented sequential-feedback path matches the
controller's first required result.

## Reliability

- Mutating Adapt calls are never blindly retried.
- A transient response to feedback or Memory storage is recorded as an
  ambiguous write and stops ordered training until reconciled.
- Safe frozen reads use bounded backoff and honor `Retry-After`.
- Each run flushes an append-only event after every confirmed transition.
- Decision-started, decision-prepared, and action-prepared WAL events preserve
  exact prompts and model inputs around partial failures.
- Checkpoints identify the last confirmed step and ambiguous interaction.
- Adapt-enabled runs never silently degrade to baseline.
- API keys and bearer strings are recursively redacted from manifests/events.

Run data lives at:

```text
.fle/adapt1/runs/<run_id>/
  manifest.json
  events.jsonl
  checkpoint.json
```

## Deterministic smoke test

This exercises the complete local FLE path without a model or Adapt key:

```bash
adapt1-fle run \
  --mode baseline \
  --env-id iron_ore_throughput \
  --steps 1 \
  --static-policy
```

The static action is `print(inspect_inventory())`. It is a diagnostic, not a
performance benchmark.

## Development

```bash
ruff check .
ruff format --check .
mypy src/adapt1_fle
pytest
```

Live tests are opt-in and require external credentials. Local Factorio tests
require the cluster started by `.cursor/start.sh`.

See:

- [Domain design](docs/DOMAIN_DESIGN.md)
- [Experiment methodology](docs/EXPERIMENTS.md)
- [Operations and recovery](docs/OPERATIONS.md)

## Research integrity and limitations

- Adapt support/confidence is not treated as calibrated correctness.
- Fallback/controller behavior is logged separately from learned Adapt behavior.
- A stored write is not treated as proof of useful learner admission.
- The current text path does not require FLE sprites; vision is deferred to a
  controlled multimodal ablation.
- Shared-Domain parallel writes and multi-agent training are intentionally
  disabled.
- The harness is designed to measure and improve a contender. It does not claim
  to be "the best" until repeated, matched-condition benchmark evidence supports
  that conclusion.

# Operations and recovery

## Preflight

```bash
source .venv/bin/activate
fle cluster start -n 1
adapt1-fle doctor
```

For an Adapt-enabled run, require:

- public health and version success;
- `credential_present: true`;
- `authentication_ok: true`;
- the selected model provider credential;
- the configured FLE task in the registry;
- local Factorio RCON at `127.0.0.1:27000`;
- a writable ledger parent.

Do not interpret `doctor`'s public health success as authenticated readiness.

## Ordered writes

Training uses one process and one writer per Domain. Do not run two `train`
commands against the same Domain ID. Parallel evaluation is safe only when:

- all calls are frozen and contractually non-mutating; or
- each worker receives a separate Domain ID.

## HTTP classification

| Status | Handling |
|---|---|
| 200-299 | Confirm and checkpoint |
| 400 / 422 | Correct payload; do not retry |
| 401 / 403 | Correct credential/ownership; do not retry |
| 404 | Correct route/resource; Domain create is allowed only in training/explicit create |
| 409 | Resolve state conflict |
| 413 | Reduce payload |
| 429 | Honor `Retry-After`; safe reads retry, writes remain unresolved |
| 502 / 503 / 504 | Safe reads use bounded backoff; write outcome is ambiguous |

The client retries safe reads up to the configured bound. It never automatically
replays Domain creation, events, feedback, or Memory storage after a transient
response.

## Ambiguous write

When feedback or Memory storage receives a timeout/transient response:

1. The interaction is appended with `ambiguous: true`.
2. The checkpoint records `ambiguous_interaction`.
3. Ordered training stops.
4. Inspect Domain/Memory state and the stable event/interaction IDs.
5. Resume only from the last confirmed step.

Do not delete the ledger, repeat the write blindly, or skip ahead.

## Domain lifecycle

```bash
adapt1-fle domain status
adapt1-fle domain create
adapt1-fle domain delete --yes
```

Deletion removes Domain evidence and learned state. It is intentionally guarded
by `--yes`. It does not clear all Adapt Memory state.

Use a new Domain ID when changing:

- observation field meaning or normalization;
- high-level action vocabulary;
- event boundary;
- reward semantics;
- grouping or evidence scope;
- result contract.

## Reset scope

Adapt routes own different state:

- `/api/v1/memory/clear`: Memory evidence/indexes;
- `/domains/{id}/clear`: Domain evidence;
- `/domains/{id}/policy/reset`: feedback policy/context;
- `/domains/{id}/adapt/reset`: interval policy only (unused here);
- Domain delete: definition and all Domain learner state.

Do not substitute one reset for another. This harness does not automatically
reset server-side state.

## Ledger recovery

Each run directory contains:

- `manifest.json`: immutable information condition;
- `events.jsonl`: monotonically sequenced append-only events;
- `checkpoint.json`: last confirmed interaction.

Opening a ledger validates every sequence number. A mismatch raises
`LedgerCorruptionError`; do not truncate or renumber evidence manually.

Curriculum resume scans manifests and accepts a job as complete only when its
event stream contains `success` or `trajectory_limit` completion. Failed and
ambiguous runs are blocked, not pending: an operator must reconcile remote
learner state before intentionally starting a replacement experiment.

## Common failures

### `string indices must be integers`

FLE 0.4.x can serialize empty `research.progress` as the string `"None"` while
its deserializer expects a sequence. The runner normalizes only the known
research fields on reset and step. Do not globally replace `"None"` because it
may be legitimate output text.

### Gym observation-space warnings

FLE 0.4.3 uses legacy Gym and emits NumPy/observation-space warnings. Reset and
step behavior is validated independently. Treat a new exception or changed
payload as a real failure; do not suppress it merely because warnings are known.

### Empty vision

Text/structured runs do not require sprites. Install them before a multimodal
experiment:

```bash
fle sprites
```

Keep vision results in a separate benchmark arm.

### Missing Adapt credential

Set one of:

```text
ADAPT1_API_KEY
REI_API_KEY
REI_SECRET_TOKEN
REI_UNIT_API_KEY
```

The harness never prints the value. Adapt-enabled commands fail closed if no
supported key is visible to the process.

### Missing model credential

The default `claude-*` model requires `ANTHROPIC_API_KEY`. OpenRouter model
names containing `/` require `OPEN_ROUTER_API_KEY`. Baseline static smoke does
not require a model key.

## Data handling

- `.env`, `.fle/`, reports, and generated artifacts are gitignored.
- Request headers are not persisted.
- Credential-bearing keys and bearer strings are recursively redacted.
- Token usage metrics are preserved; their field names are not mistaken for
  authentication tokens.
- Do not put API keys in YAML, command arguments, prompts, or Domain metadata.

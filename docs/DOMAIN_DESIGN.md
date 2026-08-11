# Factorio Domain design

## Result contract

The v1 Domain answers one question:

> Which declared high-level strategy is best supported for the current public
> Factorio state under relation `advances_goal`?

It does not generate Python, supply a solved factory, encode hidden mappings, or
receive held-out benchmark labels.

## Public vocabulary

The Domain declares task, factory state, inventory, production flow, strategy,
execution, and outcome concepts in the Adapt schema vocabulary. Available
policies are declared as named hypotheses under relation `advances_goal`, and
feedback outcomes are listed in `query_templates.feedback_outcomes`. Policies:

| Policy | Public meaning |
|---|---|
| `inspect` | Reduce missing evidence about recipes, resources, geometry, or status |
| `gather` | Acquire the minimum missing bootstrap material |
| `craft` | Manually craft only what is needed to unlock automation |
| `power` | Build, fuel, connect, repair, or expand power |
| `mine` | Build, fuel, connect, repair, or expand extraction |
| `smelt` | Build, fuel, connect, repair, or expand processing |
| `logistics` | Route items, fluids, or electricity between stages |
| `assemble` | Automate intermediate or target production |
| `research` | Produce science, operate labs, or unlock technology |
| `debug` | Diagnose and repair errors, blockage, starvation, or layout |
| `optimize` | Remove measured bottlenecks and increase throughput |
| `verify` | Run and measure the completed automation |

These meanings are public task semantics, not task-specific answers.

## Compact learner state

Decision-time state contains only information available before execution:

- task key, goal, target item, and trajectory position;
- public quota/progress for throughput tasks;
- production and automated-production scores;
- inventory quantities;
- entity counts and public statuses;
- recent public production flows;
- research count;
- game ticks/time;
- previous execution error category;
- coarse public phase.

The detailed FLE observation is supplied separately to the code-synthesis model.
It is not copied wholesale into the Domain query.

## Sequential feedback

FLE satisfies Adapt's documented condition for sequential learning:

1. an action changes the state encountered next;
2. later and terminal outcomes should revise earlier choices.

Every feedback item binds:

```text
query decision_id
-> returned strategy
-> exact generated Python
-> exact FLE execution
-> public next state
-> normalized reward
-> episode and terminal boundary
```

The payload contains both `values.reward` for immediate contextual evidence and
`values.step_reward` for ordered delayed credit. Both use the same documented
public reward in v1.

`learning.credit_assignment.neutral_reward` is `0.0`, matching FLE's
no-progress intermediate steps. All other learner settings use Adapt defaults.

## Reward

For a throughput task:

```text
(current production score - previous production score) / public quota
```

For open play:

```text
tanh(automated production delta / max(abs(previous automated score), 1))
```

Terminal public success returns `1.0`. An exact execution error subtracts
`0.25`. The final value is bounded to `[-1, 1]`.

Raw FLE reward, score, automated score, and the reward rationale remain in the
operational ledger. Adapt receives only the documented normalized value.

## Selection and fallback

The adapter accepts route-defined policy scores, ranked hypotheses, or a direct
selected policy. A unique highest supported strategy is labeled
`selected_by=adapt_1`.

Ties, absent support, or abstention use a deterministic public-phase fallback.
This is labeled `selected_by=fallback`, never Adapt behavior. The baseline and
Memory-only arms use the same deterministic controller and label it
`selected_by=controller`.

## Memory

Memory queries use task, target, phase, and previous-error context. Writes are
limited to:

- public task success;
- normalized reward at least `0.2`;
- a recurring execution error category seen at least twice.

Local fingerprints prevent duplicate writes inside a process. Raw observations
are not indiscriminately stored.

Every Memory query and write is also scoped by the versioned Domain ID. This
prevents evidence from an older contract, pilot run, or another cold Domain
from leaking into a warm-frozen comparison.

## State lifecycle

- Training reuses one Domain across curriculum episodes with one writer.
- Cold-online benchmark runs use fresh unique Domain IDs.
- Warm frozen evaluation requires an existing Domain and cannot create it.
- Domain schema, grouping, action semantics, or reward-contract changes require
  a new versioned Domain ID.
- Full cold state requires Domain deletion/recreation; Memory state is separate.

## Deliberate exclusions

- The `/adapt/*` interval policy does not match categorical strategy selection.
- A custom `learning.transition` configuration is unnecessary for the first
  controller result and would require a separately validated target geometry.
- Images are not learner inputs in v1.
- Multi-agent/shared-writer state is not enabled.

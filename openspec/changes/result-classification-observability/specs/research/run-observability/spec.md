## Purpose

Provides a unified, durable outcome classification and local observability for a routed research agent run, so tool, model, and task outcomes roll up as `level × result × reason` and can be inspected from local logs and metrics without depending on an external tracing service.

## ADDED Requirements

### Requirement: Run outcome is classified on three orthogonal axes
Every tool and model invocation outcome SHALL normalize to a triple of `level`, `result`, and `reason`, where `level` is one of `tool | model | task`, `result` is one of `ok | degraded | failed`, and `reason` (meaningful only when `result` is `degraded` or `failed`) is one of `retryable-infra | permanent-config | logic | control`.

#### Scenario: transient infrastructure failure in a tool
- **WHEN** a database connection raises an `OperationalError` during a tool call
- **THEN** the outcome is classified `level=tool`, `result=failed`, `reason=retryable-infra`

#### Scenario: model API failure shares the infrastructure reason
- **WHEN** a model provider call returns HTTP 500
- **THEN** the outcome is classified `level=model`, `result=failed`, `reason=retryable-infra`

#### Scenario: logic error is not classified as infrastructure
- **WHEN** a tool call references an unknown evidence id or fails parameter validation
- **THEN** the outcome is classified `reason=logic`, distinct from `retryable-infra`

#### Scenario: control signal is not a crash
- **WHEN** a run reaches its budget or is cancelled
- **THEN** the outcome is classified `reason=control` and treated as a normal terminal, not as a failure to retry

### Requirement: Task-layer rollup is capped at four terminal outcomes with degraded as a completion modifier
The overall run SHALL expose only `completed | incomplete | failed | cancelled` as terminal outcomes. `degraded` SHALL be a modifier of `completed` that carries a degradation-layer label (`retrieval | model | task`), not a separate peer state.

#### Scenario: completed run with a degraded component
- **WHEN** all necessary chapters complete but at least one sub-component returns `degraded`
- **THEN** the run outcome is `completed` with a `degraded` modifier and a degradation-layer label

#### Scenario: incomplete run
- **WHEN** at least one necessary chapter is missing, the rest complete normally, and no permanent failure prevents continuation
- **THEN** the run outcome is `incomplete`

#### Scenario: permanent failure
- **WHEN** a `permanent-config` failure prevents the run from completing
- **THEN** the run outcome is `failed`

#### Scenario: user cancellation
- **WHEN** the run is cancelled by a user or control signal
- **THEN** the run outcome is `cancelled`, distinct from `failed`

### Requirement: Local typed event trace is persisted per run
A run SHALL append and persist a typed, per-run event trace to a stable local path that can be read directly by a debugging agent; failure events in the trace SHALL carry `level`, `result`, and `reason`.

#### Scenario: trace survives client disconnection
- **WHEN** an SSE or other streaming client disconnects mid-run
- **THEN** the persisted trace for that run still contains the events already produced before disconnection

#### Scenario: typed failure is discoverable locally
- **WHEN** a run fails
- **THEN** the run's local trace contains a typed event that locates the failing channel, its `reason`, and its terminal state, readable without any external service query

### Requirement: Metrics and trace derive from the same classification
Aggregated metrics (counts of `degraded` by layer and a `reason` distribution) SHALL be fed by the same classification events that populate the run trace.

#### Scenario: no divergence between report and trace
- **WHEN** a run records a `degraded` outcome
- **THEN** both the aggregated metrics and the run trace report the same degraded layer and `reason`

### Requirement: Route-guard waste is measurable
When a run whose expected route is `fast` is instead routed to the multi-agent path, the system SHALL record the tokens and spurious tool-call count incurred before it is interrupted as `routed_away`.

#### Scenario: waste is recorded on a mis-route
- **WHEN** a fast-expecting request is routed to the supervisor path and then interrupted by the route guard
- **THEN** the run records the wasted tokens and the count of tool calls performed before interruption

### Requirement: Infrastructure failure is distinguishable from content-quality failure
The system SHALL keep infrastructure failures and content/quality failures distinguishable in local records, so a `failed` run that is caused by infrastructure is labeled with an infrastructure `reason`, while poor-quality output is not mislabeled as infrastructure failure.

#### Scenario: same terminal, different cause
- **WHEN** two runs both end `failed`, one from a database outage and one from the model producing missing or empty evidence
- **THEN** the local records give them different `reason` values (an infrastructure reason versus `logic`), so the two causes are not conflated
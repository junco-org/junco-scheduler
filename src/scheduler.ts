import {
  DataflowDefinitionError,
  type DataflowScheduler,
  type InputSchedule,
  type NodeAvailability,
  type NodeIdType as NodeId,
  type NodeSchedule,
  type Schedule,
} from "dataflow-engine";

/**
 * A stateful scheduling policy for {@link createScheduler}.
 *
 * `State` is inferred from {@link SchedulerOptions.state}; spell it out (`createScheduler<Cursor>`)
 * when the initial value is narrower than what the policy later stores, such as `undefined` for a
 * cursor that becomes a `NodeId`. Stateless policies pass `state: undefined`.
 *
 * `N` and `T` are the graph's message and return payloads; they default to `unknown`, and are
 * inferred from annotated callbacks or from the graph the scheduler is passed to.
 */
export type SchedulerOptions<State, N = unknown, T = unknown> = {
  /**
   * The initial scheduler-owned state: the value itself, not a factory.
   *
   * It is kept by reference for the scheduler's whole life — never serialized, cloned or reset
   * between scheduling cycles — so class instances, `Map`s and closures survive as they are. A
   * shared object is shared: independently constructed schedulers need fresh state values.
   */
  readonly state: State;

  /**
   * Plan one window over the graph's current candidates. Called synchronously by the graph once per
   * `step`, and once per window of `run`, only while at least one node is available — a quiescent
   * graph never asks.
   *
   * `available` holds exactly the nodes the engine admitted (not done, trigger evaluated,
   * availability hook passed), in node insertion order, each with its candidate-local facts:
   * `status`, the live non-destructive `view`, its `trigger` and the `satisfied` rule edges. It is the
   * engine's own map; do not mutate it.
   *
   * `state` is the returned scheduler's CURRENT `state`, read on every call. `window` and `threads`
   * are the bounds the graph was constructed with (`Infinity` when unbounded).
   *
   * Return the engine's `Schedule`: `schedule[lane][turn]` entries. Turn `w` of every lane forms one
   * column; the columns run in order and each column's lanes run concurrently. At most `threads`
   * lanes, at most `window` entries per lane, no id twice within a column, and only ids from
   * `available`. Each entry names explicit reads: `triggerInputs(candidate)` reads what the trigger
   * fired on, while a hand-built map such as `new Map([[edgeId, 2]])` reads at most two messages.
   * A ready node's first occurrence only primes it and reads nothing. Return `[]` only for an empty
   * `available`; an empty plan while work remains is a definition error. The returned plan is used
   * as is — no normalization or copy.
   */
  readonly schedule: (
    available: ReadonlyMap<NodeId, NodeAvailability<N, T>>,
    state: State,
    window: number,
    threads: number,
  ) => Schedule;

  /**
   * Re-plan the reads of one occurrence of an already-running node, from `live`, the candidate's
   * availability snapshot taken just before its column runs. Called with the scheduler's CURRENT
   * `state`, never for a prime.
   *
   * MUST be synchronous and pure: no graph mutation, no mutation of `live`. Its result is validated
   * like a planned read map; an exception rejects the column before any node advances. Omit it to
   * keep every planned `inputs` map unchanged.
   */
  readonly resolveInputs?: (
    entry: NodeSchedule,
    live: NodeAvailability<N, T>,
    state: State,
  ) => InputSchedule;
};

/**
 * Build a dataflow-engine scheduler from a policy function and its own mutable state.
 *
 * The result implements the engine's `DataflowScheduler` and exposes {@link SchedulerOptions.state}
 * as its own mutable `state` property. Every callback receives the CURRENT `scheduler.state`, so:
 * - mutating a state object (from a callback or from outside) persists into later cycles;
 * - assigning `scheduler.state = next` replaces it for every later call, primitive state included.
 *
 * One scheduler configures exactly one graph: the graph constructor calls `create(window, threads)`
 * once, the bounds are kept and passed to `schedule`. Constructing a second graph with the same
 * scheduler throws `DataflowDefinitionError("Scheduler already configures a graph")`; planning
 * before any graph configured it throws `DataflowDefinitionError("Scheduler is not configured")`.
 * Policy exceptions propagate unchanged; nothing is retried, repaired or rolled back.
 *
 * @example Full parallel: every candidate in its own lane, one turn per window.
 * ```ts
 * import { DataflowGraph, triggerInputs, UNBOUNDED, Width } from "dataflow-engine";
 * import { createScheduler } from "junco-scheduler";
 *
 * const parallel = createScheduler({
 *   state: undefined,
 *   schedule: (available) =>
 *     [...available].map(([id, candidate]) => [{ id, inputs: triggerInputs(candidate) }]),
 * });
 * const graph = new DataflowGraph<void, number, string, never>(parallel, Width(1), UNBOUNDED);
 * ```
 *
 * @example Round robin: one occurrence per window, rotating by id from a mutable cursor.
 * ```ts
 * import { type NodeIdType, triggerInputs } from "dataflow-engine";
 * import { createScheduler } from "junco-scheduler";
 *
 * const rotation = createScheduler<{ last: NodeIdType | undefined }>({
 *   state: { last: undefined },
 *   schedule(available, state) {
 *     let first: NodeIdType | undefined;
 *     let next: NodeIdType | undefined;
 *     for (const id of available.keys()) {
 *       if (first === undefined || id < first) first = id;
 *       if (state.last !== undefined && id > state.last && (next === undefined || id < next)) next = id;
 *     }
 *     const chosen = next ?? first;
 *     if (chosen === undefined) return [];
 *     state.last = chosen; // persists: the same object is passed to the next cycle
 *     return [[{ id: chosen, inputs: triggerInputs(available.get(chosen)!) }]];
 *   },
 * });
 * rotation.state.last; // the cursor, readable and writable between steps
 * ```
 *
 * @example Fixed-size batches: plan two messages per occurrence, re-planned on the live queue.
 * ```ts
 * const pairs = createScheduler({
 *   state: { size: 2 },
 *   schedule: (available, state) =>
 *     [...available].map(([id, candidate]) => [
 *       { id, inputs: new Map(candidate.trigger.rule.map((edge) => [edge, state.size])) },
 *     ]),
 *   resolveInputs: (_entry, live, state) => new Map(live.trigger.rule.map((edge) => [edge, state.size])),
 * });
 * ```
 *
 * @example An LLM scheduler: Pi's ready-made Opus 5.5 driver from `junco-scheduler/agent`.
 * Its synchronous `schedule` only consumes the decision its native `schedule` tool staged; the
 * model itself is driven through the Agentic iterator, never by `graph.run()`.
 * ```ts
 * import { DataflowGraph, UNBOUNDED, Width } from "dataflow-engine";
 * import { PiScheduler } from "junco-scheduler/agent";
 *
 * const scheduler = await PiScheduler.opus({ prompt: "Run available work." });
 * try {
 *   const graph = new DataflowGraph(scheduler, Width(1), UNBOUNDED); // configures and binds it
 *   // ...graph.addNode(...) / graph.addEdge(...)
 *   while (true) {
 *     const { done, value } = await scheduler.next();
 *     if (done || value.kind !== "yield") break; // settled or failed
 *   }
 * } finally {
 *   await scheduler.return();
 * }
 * ```
 */
export function createScheduler<State, N = unknown, T = unknown>(
  options: SchedulerOptions<State, N, T>,
): DataflowScheduler<N, T> & {
  /** The scheduler-owned state every callback receives; mutate it, or assign a replacement. */
  state: State;
} {
  const { schedule, resolveInputs } = options;
  let window: number | undefined;
  let threads: number | undefined;
  const scheduler: DataflowScheduler<N, T> & { state: State } = {
    state: options.state,
    create(configuredWindow, configuredThreads) {
      if (window !== undefined || threads !== undefined) {
        throw new DataflowDefinitionError("Scheduler already configures a graph");
      }
      window = configuredWindow;
      threads = configuredThreads;
    },
    schedule(available) {
      if (window === undefined || threads === undefined) {
        throw new DataflowDefinitionError("Scheduler is not configured");
      }
      return schedule(available, scheduler.state, window, threads);
    },
  };
  if (resolveInputs !== undefined) {
    scheduler.resolveInputs = (entry, live) => resolveInputs(entry, live, scheduler.state);
  }
  return scheduler;
}

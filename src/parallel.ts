import { type DataflowScheduler, type NodeSchedule, triggerInputs } from "dataflow-engine";
import { createScheduler } from "./scheduler.js";

/**
 * Run every candidate at once: one single-entry lane per available node, in candidate order, one
 * turn per window. An empty candidate set plans `[]`. Stateless (`state: undefined`).
 *
 * The plan is never capped or split across later columns, so its lane count is the number of
 * available nodes. Construct the graph with `threads = UNBOUNDED` and `window = Width(1)`: with
 * fewer finite threads than candidates the graph rejects the plan as a definition error before any
 * node advances, rather than silently throttling. Each entry reads `triggerInputs(candidate)`.
 *
 * @example
 * ```ts
 * import { DataflowGraph, UNBOUNDED, Width } from "dataflow-engine";
 * import { createParallelScheduler } from "junco-scheduler";
 *
 * const graph = new DataflowGraph<void, number, string, never>(
 *   createParallelScheduler(),
 *   Width(1),
 *   UNBOUNDED,
 * );
 * ```
 */
export function createParallelScheduler<N = unknown, T = unknown>(): DataflowScheduler<N, T> {
  return createScheduler<undefined, N, T>({
    state: undefined,
    schedule: (available) => {
      const lanes: NodeSchedule[][] = [];
      for (const [id, candidate] of available) lanes.push([{ id, inputs: triggerInputs(candidate) }]);
      return lanes;
    },
  });
}

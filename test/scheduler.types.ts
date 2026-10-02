// Compile-only: builder state and payload inference. Covered by tsconfig.typecheck.json; never executed.

import type { PromptTemplate, Skill } from "@earendil-works/pi-agent-core";
import {
  DataflowGraph,
  type DataflowScheduler,
  type EdgeIdType,
  type NodeAvailability,
  type NodeIdType,
  triggerInputs,
  UNBOUNDED,
  Width,
} from "dataflow-engine";
import type { AgenticOptions } from "junco-agent";
import { PiScheduler, type PiSchedulerOptions, type ScheduleTool } from "../src/agent.ts";
import { createParallelScheduler, createRoundRobinScheduler, createScheduler } from "../src/index.ts";

class Ledger {
  readonly seen = new Map<NodeIdType, number>();
  record(id: NodeIdType): number {
    const count = (this.seen.get(id) ?? 0) + 1;
    this.seen.set(id, count);
    return count;
  }
}

export function classStateKeepsItsMethods(edge: EdgeIdType) {
  const scheduler = createScheduler({
    state: new Ledger(),
    schedule(available, state, window, threads) {
      const capacity: number = window * threads;
      const counts: number[] = [...available.keys()].map((id) => state.record(id));
      return counts.length > capacity ? [] : [...available].map(([id, live]) => [{ id, inputs: triggerInputs(live) }]);
    },
    resolveInputs: (_entry, live, state) => {
      const ledger: Ledger = state;
      return ledger.seen.size > 0 ? triggerInputs(live) : new Map([[edge, 1]]);
    },
  });
  const count: number = scheduler.state.record("a" as NodeIdType);
  scheduler.state = new Ledger();
  // @ts-expect-error the state keeps its inferred class type
  scheduler.state = { seen: new Map() };
  // @ts-expect-error nor may it become a primitive
  scheduler.state = 1;
  return count;
}

export function payloadsFlowIntoTheCandidateView(edge: EdgeIdType) {
  const scheduler = createScheduler({
    state: { reads: 0 },
    schedule(available: ReadonlyMap<NodeIdType, NodeAvailability<number, string>>, state) {
      for (const candidate of available.values()) {
        const queued: readonly number[] | undefined = candidate.view.messages.get(edge);
        const result: string | undefined = candidate.view.values.get(edge);
        // @ts-expect-error message payloads are numbers
        const wrong: readonly string[] | undefined = candidate.view.messages.get(edge);
        state.reads += (queued?.length ?? 0) + (result === undefined ? 0 : 1) + (wrong === undefined ? 0 : 1);
      }
      return [...available].map(([id, candidate]) => [{ id, inputs: triggerInputs(candidate) }]);
    },
  });
  const typed: DataflowScheduler<number, string> & { state: { reads: number } } = scheduler;
  const graph = new DataflowGraph<void, number, string, never, typeof scheduler>(scheduler, Width(1), UNBOUNDED);
  const reads: number = graph.scheduler.state.reads;
  return [typed, reads];
}

export function shippedStrategiesKeepTheirShapes() {
  const parallel = createParallelScheduler<number, string>();
  // @ts-expect-error the stateless strategy exposes no state
  parallel.state;
  const rotation = createRoundRobinScheduler<number, string>();
  const cursor: NodeIdType | undefined = rotation.state;
  rotation.state = undefined;
  // @ts-expect-error the cursor is a branded NodeId, not a raw string
  rotation.state = "a";
  const graph = new DataflowGraph<void, number, string, never, typeof rotation>(rotation, Width(1), Width(1));
  const last: NodeIdType | undefined = graph.scheduler.state;
  return [parallel, cursor, last];
}

export function bothFactoriesShareOneOptionsShape<N, T, S extends Skill, P extends PromptTemplate>(
  shared: PiSchedulerOptions<N, T, S, P>,
  native: Required<Pick<AgenticOptions<S, P, ScheduleTool>, "model" | "models" | "session">>,
) {
  const empty: PiSchedulerOptions = {};
  const made: Promise<PiScheduler<N, T, S, P>> = PiScheduler.make(shared);
  const opus: Promise<PiScheduler<N, T, S, P>> = PiScheduler.opus(shared);
  const resolved: PiSchedulerOptions<N, T, S, P> = { ...native, prompt: "Run available work." };
  const fromNative: Promise<PiScheduler<N, T, S, P>> = PiScheduler.make(resolved);
  // A redundant model is accepted on opus, which keeps its own fixed selection.
  const opusWithModel: Promise<PiScheduler<N, T, S, P>> = PiScheduler.opus(resolved);
  const unchecked: Promise<PiScheduler> = PiScheduler.make({});
  const defaulted: Promise<PiScheduler> = PiScheduler.opus();
  return [empty, made, opus, fromNative, opusWithModel, unchecked, defaulted];
}

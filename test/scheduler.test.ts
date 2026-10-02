import { test } from "bun:test";
import assert from "node:assert/strict";
import {
  always,
  DataflowDefinitionError,
  DataflowGraph,
  type EdgeIdType,
  emitOnce,
  from,
  type MessageSchedule,
  messageEdge,
  msg,
  type NodeAvailability,
  NodeId,
  type NodeIdType,
  type Schedule,
  type StepFn,
  stepNode,
  triggerInputs,
  UNBOUNDED,
  type WidthType,
  Width,
  when,
} from "dataflow-engine";
import { createParallelScheduler, createRoundRobinScheduler, createScheduler } from "../src/index.ts";

type Graph = DataflowGraph<void, number, string, never>;

/** An actor that never finishes and records every activation, its prime included. */
const actor =
  (ran: string[]): StepFn<void, number, string> =>
  async function* (this) {
    while (true) {
      ran.push(this.id);
      yield [];
    }
  };

/** A graph over `scheduler` with one always-available actor per id, in the given insertion order. */
function actors(
  scheduler: ConstructorParameters<typeof DataflowGraph<void, number, string, never>>[0],
  ids: readonly string[],
  { window = Width(1), threads = UNBOUNDED }: { readonly window?: WidthType; readonly threads?: WidthType } = {},
) {
  const ran: string[] = [];
  const graph: Graph = new DataflowGraph(scheduler, window, threads);
  for (const id of ids) graph.addNode({ id: NodeId(id), state: undefined, trigger: always(), step: actor(ran) });
  return { graph, ran };
}

/** Synthetic always-triggered candidates, for direct planning calls after `create`. */
const candidates = (...ids: readonly string[]): ReadonlyMap<NodeIdType, NodeAvailability<number, string>> =>
  new Map(
    ids.map((id) => [
      NodeId(id),
      { view: { messages: new Map(), values: new Map() }, trigger: always(), satisfied: [], status: "running" },
    ]),
  );

const planned = (schedule: Schedule): string[][] => schedule.map((lane) => lane.map((entry) => entry.id));

/** Prefers one node by id; remembers how often it planned each node. */
class Preference {
  readonly runs = new Map<NodeIdType, number>();
  constructor(public favorite: NodeIdType) {}

  pick(available: ReadonlyMap<NodeIdType, unknown>): NodeIdType {
    return available.has(this.favorite) ? this.favorite : available.keys().next().value!;
  }
}

const prefer = (available: ReadonlyMap<NodeIdType, NodeAvailability<number, string>>, state: Preference): Schedule => {
  const id = state.pick(available);
  state.runs.set(id, (state.runs.get(id) ?? 0) + 1);
  return [[{ id, inputs: triggerInputs(available.get(id)!) }]];
};

test("mutating a class state steers later cycles, and fresh states stay independent", async () => {
  const initial = new Preference(NodeId("a"));
  const first = createScheduler({ state: initial, schedule: prefer });
  const second = createScheduler({ state: new Preference(NodeId("a")), schedule: prefer });
  const one = actors(first, ["a", "b"]);
  const two = actors(second, ["a", "b"]);

  await one.graph.step();
  first.state.favorite = NodeId("b");
  await one.graph.step();
  await two.graph.step();
  await two.graph.step();

  assert.equal(first.state, initial);
  assert.ok(first.state instanceof Preference);
  assert.deepEqual(one.ran, ["a", "b"]);
  assert.deepEqual([...first.state.runs], [["a", 1], ["b", 1]]);
  assert.deepEqual(two.ran, ["a", "a"]);
  assert.deepEqual([...second.state.runs], [["a", 2]]);
});

test("a replaced state drives both selection and live reads: explicit pairs deliver [1,2], [3,4], [5]", async () => {
  type Reads = { readonly target: NodeIdType; readonly size: number };
  const feed = messageEdge<number, string>("feed");
  const seen: Reads[] = [];
  const scheduler = createScheduler<Reads, number, string>({
    state: { target: NodeId("P"), size: 1 },
    schedule(available, state) {
      seen.push(state);
      // Plans whole-queue reads; only the live resolver sizes a running occurrence.
      return [[{ id: state.target, inputs: triggerInputs(available.get(state.target)!) }]];
    },
    resolveInputs(_entry, live, state) {
      seen.push(state);
      return new Map<EdgeIdType, MessageSchedule>(live.trigger.rule.map((edge) => [edge, state.size]));
    },
  });
  const graph: Graph = new DataflowGraph(scheduler, Width(1), UNBOUNDED);
  const p = graph.addNode({
    id: NodeId("P"),
    state: undefined,
    trigger: always(),
    step: emitOnce(() => [1, 2, 3, 4, 5].map((n) => msg(feed, n)), "p"),
  });
  const batches: number[][] = [];
  const c = graph.addNode({
    id: NodeId("C"),
    state: undefined,
    trigger: when(feed),
    step: stepNode((_state, view) => {
      batches.push([...from(view, feed)]);
      return [];
    }),
  });
  graph.addEdge(p, c, feed);

  await graph.step(); // primes P, which emits the whole feed
  const replacement: Reads = { target: NodeId("C"), size: 2 };
  scheduler.state = replacement;
  for (let window = 0; window < 4; window++) await graph.step(); // primes C, then three reads

  assert.deepEqual(batches, [[1, 2], [3, 4], [5]]);
  assert.equal(graph.getNode(p)!.status, "running");
  assert.equal(seen[0]!.target, NodeId("P"));
  assert.ok(seen.slice(1).every((state) => state === replacement));
  // Window 1 planned P; windows 2-5 each planned C, and every running occurrence was re-read live.
  assert.equal(seen.length, 1 + 4 + 3);
});

test("full parallel starts every candidate in one column and advances each exactly once", async () => {
  const gate = Promise.withResolvers<void>();
  const allStarted = Promise.withResolvers<void>();
  const started: string[] = [];
  const advanced = new Map<string, number>();
  const graph: Graph = new DataflowGraph(createParallelScheduler(), Width(1), UNBOUNDED);
  for (const id of ["a", "b", "c"]) {
    graph.addNode({
      id: NodeId(id),
      state: undefined,
      trigger: always(),
      step: async function* () {
        started.push(id);
        if (started.length === 3) allStarted.resolve();
        await gate.promise;
        advanced.set(id, (advanced.get(id) ?? 0) + 1);
        while (true) yield [];
      },
    });
  }

  const stepping = graph.step();
  await allStarted.promise; // every lane is in flight while all gates are still closed
  assert.deepEqual(started, ["a", "b", "c"]);
  assert.equal(advanced.size, 0);
  gate.resolve();
  await stepping;

  assert.deepEqual([...advanced], [["a", 1], ["b", 1], ["c", 1]]);
  for (const id of ["a", "b", "c"]) assert.equal(graph.getNode(NodeId(id))!.status, "running");
});

test("full parallel plans nothing for no candidates and never throttles finite threads", async () => {
  const empty = createParallelScheduler<number, string>();
  empty.create(1, Number.POSITIVE_INFINITY);
  assert.deepEqual(empty.schedule(candidates()), []);
  assert.deepEqual(planned(empty.schedule(candidates("x", "y"))), [["x"], ["y"]]);

  const { graph, ran } = actors(createParallelScheduler(), ["a", "b", "c"], { threads: Width(2) });
  await assert.rejects(graph.step(), DataflowDefinitionError);
  assert.deepEqual(ran, []);
  for (const id of ["a", "b", "c"]) assert.equal(graph.getNode(NodeId(id))!.status, "ready");
});

test("round robin runs one occurrence per window in string order: 1, 10, 2, 1", async () => {
  const scheduler = createRoundRobinScheduler<number, string>();
  const { graph, ran } = actors(scheduler, ["2", "1", "10"], { window: Width(8), threads: Width(8) });

  for (let window = 0; window < 4; window++) await graph.step();

  assert.deepEqual(ran, ["1", "10", "2", "1"]);
  assert.equal(scheduler.state, NodeId("1"));
});

test("round robin resumes after its cursor, wraps only past the largest id, and keeps the cursor when idle", () => {
  const scheduler = createRoundRobinScheduler<number, string>();
  scheduler.create(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY);
  const plan = (...ids: string[]) => planned(scheduler.schedule(candidates(...ids)));

  scheduler.state = NodeId("b"); // the previous id vanished: its smallest successor runs
  assert.deepEqual(plan("a", "c"), [["c"]]);
  scheduler.state = NodeId("z");
  assert.deepEqual(plan("a", "c"), [["a"]]);

  assert.deepEqual(plan(), []);
  assert.equal(scheduler.state, NodeId("a"));

  assert.deepEqual(plan("b", "c"), [["b"]]);
  assert.deepEqual(plan("a", "b", "c"), [["c"]]); // a new smaller candidate waits for the wrap
  assert.deepEqual(plan("a", "b", "c"), [["a"]]);

  scheduler.state = NodeId("");
  assert.deepEqual(plan("", "a"), [["a"]]); // "" is a cursor, not "uninitialized"
  assert.deepEqual(plan("", "a"), [[""]]);
  scheduler.state = undefined;
  assert.deepEqual(plan("b", "", "a"), [[""]]);
});

test("a scheduler configures exactly one graph and refuses to plan unconfigured", () => {
  const unconfigured = createScheduler({ state: undefined, schedule: () => [] });
  assert.throws(
    () => unconfigured.schedule(candidates("a")),
    (error) => error instanceof DataflowDefinitionError && error.message === "Scheduler is not configured",
  );

  const scheduler = createRoundRobinScheduler<number, string>();
  actors(scheduler, ["a"]);
  assert.throws(
    () => new DataflowGraph(scheduler, Width(1), UNBOUNDED),
    (error) => error instanceof DataflowDefinitionError && error.message === "Scheduler already configures a graph",
  );
});

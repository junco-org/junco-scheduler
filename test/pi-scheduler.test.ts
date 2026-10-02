import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type AgentTool, BACKGROUND_CONTEXT, MemorySessionRepo } from "@earendil-works/pi-agent-core";
import {
  type Context,
  createModels,
  fauxAssistantMessage,
  fauxProvider,
  InMemoryCredentialStore,
} from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import {
  always,
  DataflowDefinitionError,
  DataflowGraph,
  DataflowNodeError,
  type NodeGenerator,
  NodeId,
  never,
  UNBOUNDED,
  Width,
} from "dataflow-engine";
import { JuncoAgentError } from "junco-agent";
import { Type } from "typebox";
import { PiScheduler, type PiSchedulerOptions, type SchedulerInfo } from "../src/agent.ts";
import {
  drive,
  nativeOptions,
  newScheduler,
  runAvailable,
  type Script,
  scheduleCall,
  schedulerInfoOf,
  settledText,
  toolErrors,
  toolTurn,
} from "./support.ts";

type Graph = DataflowGraph<void, string, string, unknown, PiScheduler<string, string>>;

/** A node that primes, resumes once, then returns `result`, logging each activation. */
const twoStep = (log: string[], result: string) =>
  async function* (): NodeGenerator<string, string> {
    log.push("prime");
    yield [];
    log.push("resume");
    return result;
  };

/** A graph over a fresh scheduler with one two-step node `"0"`. */
async function singleNode(script: Script, result = "node result") {
  const { scheduler, faux, session } = await newScheduler<string, string>(script, { prompt: "schedule it" });
  const graph: Graph = new DataflowGraph(scheduler, UNBOUNDED, UNBOUNDED);
  const log: string[] = [];
  graph.addNode({ id: NodeId("0"), state: undefined, trigger: always(), step: twoStep(log, result) });
  return { scheduler, graph, faux, session, log };
}

const unbound = (error: unknown) =>
  error instanceof JuncoAgentError &&
  error.code === "invalid_state" &&
  error.message === "PiScheduler is not bound to a graph";

test("the empty node id primes, resumes and completes through real schedule tool calls", async () => {
  const { scheduler } = await newScheduler<string, string>(
    [
      toolTurn(scheduleCall([[""]], "prime")),
      toolTurn(scheduleCall([[""]], "resume")),
      toolTurn(scheduleCall([[""]], "complete")),
      fauxAssistantMessage("done"),
    ],
    { prompt: "run it" },
  );
  const graph: Graph = new DataflowGraph(scheduler, Width(1), UNBOUNDED);
  const log: string[] = [];
  graph.addNode({
    id: NodeId(""),
    state: undefined,
    trigger: always(),
    step: async function* () {
      log.push("prime");
      yield [];
      log.push("resume");
      yield [];
      log.push("complete");
      return "empty-id result";
    },
  });
  const errors = toolErrors(scheduler);
  try {
    const { yields, last } = await drive(scheduler);
    assert.equal(settledText(last), "done");
    assert.equal(yields, 3);
    assert.deepEqual(errors, [false, false, false]);
    assert.deepEqual(log, ["prime", "resume", "complete"]);
    assert.deepEqual(graph.result(NodeId("")), { _tag: "Some", value: "empty-id result" });
  } finally {
    await scheduler.return();
  }
});

test("the fixed schedule tool cannot be replaced or disabled", async () => {
  const { scheduler } = await newScheduler([]);
  const [tool] = scheduler.getTools();
  assert.equal(scheduler.getTools().length, 1);
  assert.equal(tool!.name, "schedule");

  const other: AgentTool = {
    name: "other",
    label: "other",
    description: "not allowed",
    parameters: Type.Object({}),
    execute: async () => ({ content: [], details: undefined }),
  };
  const invalid = (promise: Promise<void>) =>
    assert.rejects(promise, (error) => error instanceof JuncoAgentError && error.code === "invalid_argument");
  await invalid(scheduler.setTools([other as never]));
  await invalid(scheduler.setTools([tool!, tool!]));
  await invalid(scheduler.setTools([tool!], ["other"]));
  await invalid(scheduler.setTools([]));
  await invalid(scheduler.setTools([tool!], []));
  await invalid(scheduler.setActiveTools([]));
  assert.equal(scheduler.getTools()[0], tool);
  await scheduler.setTools([tool!], ["schedule"]);
  await scheduler.setActiveTools(["schedule"]);
  assert.deepEqual(
    scheduler.getActiveTools().map((active) => active.name),
    ["schedule"],
  );
  await scheduler.return();
});

test("each schedule call runs one real graph window and pauses until the next resume", async () => {
  const { scheduler, graph, faux, log } = await singleNode(
    [toolTurn(scheduleCall([["0"]], "prime")), toolTurn(scheduleCall([["0"]], "run")), fauxAssistantMessage("finished")],
    "actual node result",
  );
  const completed: SchedulerInfo[] = [];
  scheduler.events.on("tool_end", ({ toolName, result, isError }) => {
    if (toolName === "schedule" && !isError) completed.push(result.details as SchedulerInfo);
  });
  try {
    const first = await scheduler.next();
    assert.ok(first.done === false && first.value.kind === "yield");
    assert.deepEqual(completed[0]!.nodes, [{ id: "0", status: "running" }]);
    assert.equal(graph.getNode(NodeId("0"))!.status, "running");
    await new Promise<void>((resolve) => setImmediate(resolve)); // nothing progresses while paused
    assert.equal(faux.state.callCount, 1);

    const second = await scheduler.next();
    assert.ok(second.done === false && second.value.kind === "yield");
    assert.deepEqual(completed[1], { window: null, threads: null, quiescent: true, nodes: [] });
    assert.equal(graph.getNode(NodeId("0"))!.status, "done");
    assert.equal(faux.state.callCount, 2);
    assert.equal(settledText((await drive(scheduler)).last), "finished");
    assert.equal(faux.state.callCount, 3);
    assert.deepEqual(log, ["prime", "resume"]);
    assert.deepEqual(graph.result(NodeId("0")), { _tag: "Some", value: "actual node result" });
  } finally {
    await scheduler.return();
  }
});

test("schedule calls added by a later after_response hook are rejected before any graph step", async () => {
  let graphSteps = 0;
  const stepsAtRetry: number[] = [];
  const statusAtRetry: string[] = [];
  const { scheduler, graph, log } = await singleNode([
    toolTurn(scheduleCall([["0"]], "preliminary")),
    (context: Context) => {
      statusAtRetry.push(schedulerInfoOf(context).nodes[0]!.status);
      stepsAtRetry.push(graphSteps);
      return toolTurn(scheduleCall([["0"]], "prime"));
    },
    toolTurn(scheduleCall([["0"]], "run")),
    fauxAssistantMessage("finished"),
  ]);
  const step = graph.step.bind(graph);
  graph.step = (options) => {
    graphSteps++;
    return step(options);
  };
  scheduler.hooks.on("after_response", ({ message }) =>
    message.content.some((block) => block.type === "toolCall" && block.id === "preliminary")
      ? { message: { ...message, content: [scheduleCall([["0"]], "a"), scheduleCall([["0"]], "b")] } }
      : undefined,
  );
  const errors = toolErrors(scheduler);

  const { yields, last } = await drive(scheduler);

  assert.deepEqual(errors, [true, true, false, false]);
  assert.deepEqual(statusAtRetry, ["ready"]);
  assert.deepEqual(stepsAtRetry, [0]);
  assert.equal(graphSteps, 2);
  assert.equal(yields, 2);
  assert.equal(settledText(last), "finished");
  assert.deepEqual(log, ["prime", "resume"]);
  await scheduler.return();
});

test("an invalid plan is an ordinary tool error that activates nothing", async () => {
  const statusAfter: string[] = [];
  const { scheduler, log } = await singleNode([
    toolTurn(scheduleCall([["missing"]], "bad")),
    (context: Context) => {
      statusAfter.push(schedulerInfoOf(context).nodes[0]!.status);
      return toolTurn(scheduleCall([["0"]], "prime"));
    },
    toolTurn(scheduleCall([["0"]], "run")),
    fauxAssistantMessage("finished"),
  ]);
  const errors = toolErrors(scheduler);

  assert.equal(settledText((await drive(scheduler)).last), "finished");
  assert.deepEqual(errors, [true, false, false]);
  assert.deepEqual(statusAfter, ["ready"]);
  assert.deepEqual(log, ["prime", "resume"]);
  await scheduler.return();
});

test("with nothing available a plan is rejected, while an empty one returns SchedulerInfo without a pause", async () => {
  const { scheduler } = await newScheduler<string, string>(
    [toolTurn(scheduleCall([["0"]], "work")), toolTurn(scheduleCall([], "empty")), fauxAssistantMessage("nothing to do")],
    { prompt: "go" },
  );
  const graph: Graph = new DataflowGraph(scheduler, UNBOUNDED, UNBOUNDED);
  const log: string[] = [];
  graph.addNode({ id: NodeId("0"), state: undefined, trigger: never(), step: twoStep(log, "never") });
  const errors = toolErrors(scheduler);
  const results: SchedulerInfo[] = [];
  scheduler.events.on("tool_end", ({ result, isError }) => {
    if (!isError) results.push(result.details as SchedulerInfo);
  });

  const { yields, last } = await drive(scheduler);
  assert.equal(yields, 0);
  assert.equal(settledText(last), "nothing to do");
  assert.deepEqual(errors, [true, false]);
  assert.deepEqual(results, [{ window: null, threads: null, quiescent: true, nodes: [] }]);
  assert.deepEqual(log, []);
  await scheduler.return();
});

test("finishing while work is available fails the run, before or after a window", async () => {
  const early = await singleNode([fauxAssistantMessage("done already")]);
  const skipped = (await drive(early.scheduler)).last;
  assert.ok(skipped.kind === "failed" && skipped.error instanceof JuncoAgentError);
  assert.equal(skipped.error.code, "invalid_state");

  const primedOnly = await singleNode([toolTurn(scheduleCall([["0"]], "prime")), fauxAssistantMessage("done")]);
  const { yields, last } = await drive(primedOnly.scheduler);
  assert.equal(yields, 1);
  assert.ok(last.kind === "failed" && last.error instanceof JuncoAgentError && last.error.code === "invalid_state");
  await early.scheduler.return();
  await primedOnly.scheduler.return();
});

test("a node failure remains fatal even after the scheduler's final answer", async () => {
  const cause = new Error("node input failed");
  const { scheduler } = await newScheduler<string, string>(
    [toolTurn(scheduleCall([["0"]], "prime")), toolTurn(scheduleCall([["0"]], "run")), fauxAssistantMessage("all good")],
    { prompt: "go" },
  );
  const graph: Graph = new DataflowGraph(scheduler, UNBOUNDED, UNBOUNDED);
  graph.addNode({
    id: NodeId("0"),
    state: undefined,
    trigger: always(),
    step: async function* () {
      yield [];
      throw cause;
    },
  });

  const { last } = await drive(scheduler);
  assert.ok(last.kind === "failed" && last.error instanceof DataflowNodeError);
  assert.equal(last.error.nodeId, NodeId("0"));
  assert.equal(last.error.error, cause);
  await scheduler.return();
});

test("a scheduler binds one graph, and an unstaged step fails only while work is available", async () => {
  const { scheduler } = await newScheduler<string, string>([]);
  assert.throws(
    () => scheduler.bind(() => new Map(), async () => {}),
    (error) => error instanceof DataflowDefinitionError && error.message === "PiScheduler is not configured",
  );
  const graph: Graph = new DataflowGraph(scheduler, UNBOUNDED, UNBOUNDED);
  const log: string[] = [];
  graph.addNode({ id: NodeId("0"), state: undefined, trigger: always(), step: twoStep(log, "never") });

  await assert.rejects(
    graph.step(),
    (error) =>
      error instanceof DataflowDefinitionError && error.message === "PiScheduler requires a schedule-tool decision",
  );
  assert.equal(graph.getNode(NodeId("0"))!.status, "ready");
  assert.deepEqual(log, []);
  assert.throws(() => new DataflowGraph(scheduler, UNBOUNDED, UNBOUNDED), DataflowDefinitionError);
  assert.throws(
    () => scheduler.bind(() => new Map(), async () => {}),
    (error) => error instanceof DataflowDefinitionError && error.message === "PiScheduler already binds a graph",
  );

  const idle = await newScheduler<string, string>([]);
  const quiet: Graph = new DataflowGraph(idle.scheduler, UNBOUNDED, UNBOUNDED);
  quiet.addNode({ id: NodeId("0"), state: undefined, trigger: never(), step: twoStep(log, "never") });
  assert.equal((await quiet.step()).size, 0);
  assert.deepEqual(log, []);
  await scheduler.return();
  await idle.scheduler.return();
});

test("closing the scheduler mid-window lets the running column drain and starts no later column", async () => {
  const gate = Promise.withResolvers<void>();
  const started = Promise.withResolvers<void>();
  const { scheduler } = await newScheduler<string, string>([toolTurn(scheduleCall([["A", "B"]], "window"))], {
    prompt: "go",
  });
  const graph: Graph = new DataflowGraph(scheduler, UNBOUNDED, UNBOUNDED);
  const log: string[] = [];
  graph.addNode({
    id: NodeId("A"),
    state: undefined,
    trigger: always(),
    step: async function* (): NodeGenerator<string, string> {
      started.resolve();
      await gate.promise;
      log.push("A primed");
      yield [];
      return "a";
    },
  });
  graph.addNode({ id: NodeId("B"), state: undefined, trigger: always(), step: twoStep(log, "never") });

  const running = scheduler.next();
  await started.promise;
  const closing = scheduler.return();
  gate.resolve();
  await closing;
  await running.catch(() => undefined);

  assert.equal(scheduler.status, "closed");
  assert.deepEqual(log, ["A primed"]);
  assert.equal(graph.getNode(NodeId("A"))!.status, "running");
  assert.equal(graph.getNode(NodeId("B"))!.status, "ready");
});

test("SchedulerInfo shows only candidates: no unavailable node, finished node or result reaches the model", async () => {
  const HIDDEN = "hidden-5ec7e7";
  const RESULT = "RESULT-c0ffee";
  const requests: { readonly info: SchedulerInfo; readonly transcript: string }[] = [];
  const { scheduler, session } = await newScheduler<string, string>(
    (info) => {
      requests.push({ info, transcript: "" });
      return runAvailable(info);
    },
    {
      prompt: "go",
      adapter: {
        describeNode: (id, candidate) => {
          described.push(id);
          return { trigger: candidate.trigger.kind, satisfied: candidate.satisfied.length };
        },
      },
    },
  );
  const described: string[] = [];
  const transcripts: string[] = [];
  scheduler.hooks.on("transform_context", ({ messages }) => {
    transcripts.push(JSON.stringify(messages));
    return undefined;
  });
  const results: SchedulerInfo[] = [];
  scheduler.events.on("tool_end", ({ result, isError }) => {
    if (!isError) results.push(result.details as SchedulerInfo);
  });
  const emitted: string[] = [];
  scheduler.events.on("message_end", ({ message }) => {
    emitted.push(JSON.stringify(message));
  });
  const graph: Graph = new DataflowGraph(scheduler, Width(1), UNBOUNDED);
  const log: string[] = [];
  const worker = graph.addNode({ id: NodeId("worker"), state: undefined, trigger: always(), step: twoStep(log, RESULT) });
  graph.addNode({ id: NodeId(HIDDEN), state: undefined, trigger: never(), step: twoStep(log, "never") });

  try {
    assert.equal(settledText((await drive(scheduler)).last), "graph quiescent");

    const shown = (info: SchedulerInfo) => info.nodes.map((node) => [node.id, node.status, node.details]);
    const details = { trigger: "all", satisfied: 0 };
    assert.deepEqual(requests.map(({ info }) => shown(info)), [
      [["worker", "ready", details]],
      [["worker", "running", details]],
      [],
    ]);
    assert.deepEqual(results.map(shown), [[["worker", "running", details]], []]);
    assert.ok([...requests.map(({ info }) => info), ...results].every((info) => info.window === 1 && info.threads === null));
    assert.deepEqual(
      [...requests.map(({ info }) => info.quiescent), ...results.map((info) => info.quiescent)],
      [false, false, true, false, true],
    );
    assert.ok(described.every((id) => id === "worker"));
    // The whole model-visible transcript, including tool results and earlier snapshots.
    assert.equal(transcripts.length, 3);
    for (const transcript of transcripts) {
      assert.equal(transcript.includes(HIDDEN), false);
      assert.equal(transcript.includes(RESULT), false);
    }
    assert.deepEqual(graph.result(worker), { _tag: "Some", value: RESULT });
    assert.equal(JSON.stringify(await session.findEntries(undefined, BACKGROUND_CONTEXT)).includes("SchedulerInfo:"), false);
    assert.equal(emitted.some((message) => message.includes("SchedulerInfo:")), false);
  } finally {
    await scheduler.return();
  }
});

test("opus selects GitHub Copilot claude-opus-5.5 from a saved login and owns the transferred session", async () => {
  const directory = await mkdtemp(join(tmpdir(), "junco-scheduler-login-"));
  const repo = new MemorySessionRepo();
  try {
    const authPath = join(directory, "auth.json");
    await writeFile(
      authPath,
      JSON.stringify({
        "github-copilot": {
          type: "oauth",
          access: "pi-fixture-access",
          refresh: "pi-fixture-refresh",
          expires: Number.MAX_SAFE_INTEGER,
        },
      }),
    );
    const models = await ModelRuntime.create({ authPath, modelsPath: null, refreshOnCreate: false });
    const session = await repo.create({}, BACKGROUND_CONTEXT);
    // A redundant model never overrides the fixed selection.
    const conflicting = fauxProvider({ api: "conflicting", provider: "conflicting" });
    const shared: PiSchedulerOptions = { models, session, model: conflicting.getModel() };
    const scheduler = await PiScheduler.opus(shared);
    const selected = await scheduler.getModel();
    assert.equal(selected?.provider, "github-copilot");
    assert.equal(selected?.id, "claude-opus-5.5");
    await scheduler.return();
    await assert.rejects(session.beginMutation(BACKGROUND_CONTEXT));

    const missing = await ModelRuntime.create({
      authPath: join(directory, "missing.json"),
      modelsPath: null,
      refreshOnCreate: false,
    });
    const transferred = await repo.create({}, BACKGROUND_CONTEXT);
    await assert.rejects(PiScheduler.opus({ models: missing, session: transferred }), (error: unknown) => {
      assert.ok(error instanceof JuncoAgentError);
      assert.equal(error.code, "invalid_state");
      assert.match(error.message, /github-copilot/);
      assert.match(error.message, /\/login/);
      return true;
    });
    await assert.rejects(transferred.beginMutation(BACKGROUND_CONTEXT));
  } finally {
    await repo.close(BACKGROUND_CONTEXT);
    await rm(directory, { recursive: true, force: true });
  }
});

test("make validates required shared options before adopting resources", async () => {
  const { faux, options } = await nativeOptions([fauxAssistantMessage("never")]);
  const { model, models, session } = options;
  try {
    await session.setName("still-owned", BACKGROUND_CONTEXT);
    for (const partial of [
      { models, session },
      { model, session },
      { model, models },
    ] satisfies PiSchedulerOptions[]) {
      await assert.rejects(PiScheduler.make({ ...partial, prompt: "go" }), (error: unknown) => {
        assert.ok(error instanceof JuncoAgentError);
        assert.equal(error.code, "invalid_argument");
        assert.equal(error.message, "PiScheduler.make requires model, models, and session");
        return true;
      });
    }
    assert.equal(await session.getName(BACKGROUND_CONTEXT), "still-owned");
    assert.equal(faux.state.callCount, 0);
  } finally {
    await session.close(BACKGROUND_CONTEXT);
  }
});

test("opus never falls back to a supplied model when the fixed model is missing", async () => {
  const credentials = new InMemoryCredentialStore();
  await credentials.modify("github-copilot", async () => ({
    type: "oauth",
    access: "pi-fixture-access",
    refresh: "pi-fixture-refresh",
    expires: Number.MAX_SAFE_INTEGER,
  }));
  const models = await ModelRuntime.create({ credentials, modelsPath: null, refreshOnCreate: false });
  const copilot = fauxProvider({ provider: "github-copilot", models: [{ id: "alternative" }] });
  models.registerNativeProvider({ ...copilot.provider, auth: models.getProvider("github-copilot")!.auth });
  const repo = new MemorySessionRepo();
  try {
    const session = await repo.create({}, BACKGROUND_CONTEXT);
    await assert.rejects(PiScheduler.opus({ models, session, model: copilot.getModel() }), (error: unknown) => {
      assert.ok(error instanceof JuncoAgentError);
      assert.equal(error.code, "invalid_argument");
      assert.equal(error.message, "Model not found: github-copilot/claude-opus-5.5");
      return true;
    });
    await assert.rejects(session.beginMutation(BACKGROUND_CONTEXT));
    assert.equal(copilot.state.callCount, 0);
  } finally {
    await repo.close(BACKGROUND_CONTEXT);
  }
});

test("opus over a native model collection without saved logins fails its login check", async () => {
  const copilot = fauxProvider({ provider: "github-copilot", models: [{ id: "claude-opus-5.5" }] });
  copilot.setResponses([fauxAssistantMessage("never")]);
  const models = createModels();
  models.setProvider(copilot.provider);
  const repo = new MemorySessionRepo();
  try {
    const session = await repo.create({}, BACKGROUND_CONTEXT);
    await assert.rejects(PiScheduler.opus({ models, session, prompt: "go" }), (error: unknown) => {
      assert.ok(error instanceof JuncoAgentError);
      assert.equal(error.code, "invalid_state");
      assert.match(error.message, /github-copilot/);
      assert.match(error.message, /\/login/);
      return true;
    });
    await assert.rejects(session.beginMutation(BACKGROUND_CONTEXT));
    assert.equal(copilot.state.callCount, 0);
  } finally {
    await repo.close(BACKGROUND_CONTEXT);
  }
});

test("an unbound scheduler fails its first request before any provider call, through either factory", async () => {
  const credentials = new InMemoryCredentialStore();
  await credentials.modify("github-copilot", async () => ({
    type: "oauth",
    access: "pi-fixture-access",
    refresh: "pi-fixture-refresh",
    expires: Number.MAX_SAFE_INTEGER,
  }));
  const models = await ModelRuntime.create({ credentials, modelsPath: null, refreshOnCreate: false });
  const copilot = fauxProvider({ provider: "github-copilot", models: [{ id: "claude-opus-5.5" }] });
  copilot.setResponses([fauxAssistantMessage("never")]);
  // The faux transport stands in for Copilot's; the login check still uses Copilot's own auth.
  models.registerNativeProvider({ ...copilot.provider, auth: models.getProvider("github-copilot")!.auth });
  const resolved = await nativeOptions([fauxAssistantMessage("never")]);

  for (const [faux, make] of [
    [copilot, () => PiScheduler.opus({ models, prompt: "go" })],
    [resolved.faux, () => PiScheduler.make({ ...resolved.options, prompt: "go" })],
  ] as const) {
    const scheduler = await make();
    try {
      const step = await scheduler.next();
      assert.ok(step.done === false && step.value.kind === "failed", JSON.stringify(step));
      assert.ok(unbound(step.value.error), String(step.value.error));
      assert.equal(faux.state.callCount, 0);
    } finally {
      await scheduler.return();
    }
  }
});

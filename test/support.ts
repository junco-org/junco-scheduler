import { BACKGROUND_CONTEXT, MemorySessionRepo } from "@earendil-works/pi-agent-core";
import {
  type Context,
  createModels,
  type FauxResponseFactory,
  type FauxResponseStep,
  fauxAssistantMessage,
  fauxProvider,
  fauxToolCall,
  type ToolCall,
} from "@earendil-works/pi-ai";
import type { Agentic, AgenticStep } from "junco-agent";
import { PiScheduler, type PiSchedulerOptions, type SchedulerInfo } from "../src/agent.ts";

// Offline scaffolding: every scheduler gets its own faux provider in an explicit Models collection
// and an in-memory session. Scripts read the SchedulerInfo of the request they answer.

let providers = 0;

/** A faux model serving `responses` in order, with fresh native options. */
export async function nativeOptions(responses: readonly FauxResponseStep[] = []) {
  const name = `faux-${++providers}`;
  const faux = fauxProvider({ api: name, provider: name, tokenSize: { min: 4, max: 4 } });
  faux.setResponses([...responses]);
  const models = createModels();
  models.setProvider(faux.provider);
  return {
    faux,
    options: { session: await new MemorySessionRepo().create({}, BACKGROUND_CONTEXT), models, model: faux.getModel() },
  };
}

/** Every user text of one request, in order. */
export const userTexts = (context: Context): string[] =>
  context.messages.flatMap((message) =>
    message.role !== "user"
      ? []
      : [
          typeof message.content === "string"
            ? message.content
            : message.content.map((block) => (block.type === "text" ? block.text : "")).join(""),
        ],
  );

/** The SchedulerInfo appended to the request that `context` belongs to. */
export function schedulerInfoOf(context: Context): SchedulerInfo {
  const last = userTexts(context).at(-1);
  if (last === undefined || !last.startsWith("SchedulerInfo:\n")) throw new Error("request carries no SchedulerInfo");
  return JSON.parse(last.slice("SchedulerInfo:\n".length)) as SchedulerInfo;
}

export type Decision = { readonly lanes: readonly (readonly string[])[] } | { readonly text: string };

/** Schedule every candidate in its own lane; once quiescent, report a fixed completion. */
export const runAvailable = (info: SchedulerInfo): Decision =>
  info.nodes.length > 0 ? { lanes: info.nodes.map((node) => [node.id]) } : { text: "graph quiescent" };

/** One `schedule` tool call. */
export const scheduleCall = (lanes: readonly (readonly string[])[], id: string) =>
  fauxToolCall("schedule", { lanes }, { id });

/** An assistant turn making `calls`. */
export const toolTurn = (...calls: ToolCall[]) => fauxAssistantMessage(calls, { stopReason: "toolUse" });

/** Decide one scheduler request from the SchedulerInfo it carries. */
export type Decide = (info: SchedulerInfo, request: number) => Decision;

/** A scheduler script: a SchedulerInfo-driven decision, or explicit faux responses. */
export type Script = Decide | readonly FauxResponseStep[];

/** A PiScheduler over `script`, staged with `prompt` when given. */
export async function newScheduler<N = unknown, T = unknown>(
  script: Script = runAvailable,
  extra: Omit<PiSchedulerOptions<N, T>, "session" | "models" | "model"> = {},
) {
  let requests = 0;
  const scripted = (decide: Decide): FauxResponseFactory => {
    const respond: FauxResponseFactory = (context) => {
      faux.appendResponses([respond]);
      const decision = decide(schedulerInfoOf(context), requests++);
      return "lanes" in decision
        ? toolTurn(scheduleCall(decision.lanes, `schedule-${requests}`))
        : fauxAssistantMessage(decision.text);
    };
    return respond;
  };
  const { faux, options } = await nativeOptions(typeof script === "function" ? [scripted(script)] : script);
  const scheduler = await PiScheduler.make<N, T>({ ...options, ...extra });
  return { scheduler, faux, session: options.session };
}

/** Resume through every pause until the current run settles or fails; returns that step. */
export async function drive(agentic: Agentic): Promise<{ readonly yields: number; readonly last: AgenticStep }> {
  let yields = 0;
  while (true) {
    const result = await agentic.next();
    if (result.done) throw new Error("iterator finished before its run settled");
    if (result.value.kind !== "yield") return { yields, last: result.value };
    yields++;
  }
}

/** The text of a settled step. */
export function settledText(step: AgenticStep): string {
  if (step.kind !== "settled") throw new Error(`expected a settled step, got ${step.kind}`);
  return step.message.content.map((block) => (block.type === "text" ? block.text : "")).join("");
}

/** Whether each finished tool call failed, in completion order. */
export function toolErrors(agentic: Agentic): boolean[] {
  const errors: boolean[] = [];
  agentic.events.on("tool_end", ({ isError }) => {
    errors.push(isError);
  });
  return errors;
}

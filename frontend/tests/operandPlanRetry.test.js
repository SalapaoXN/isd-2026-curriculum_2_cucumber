import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {askQuestion} from "../src/api.js";
import {applyChatResponse, buildConversationContext, clarificationPlans, selectPlanForRetry} from "../src/chatScope.js";

const programs = [
  {program_code: "IT", editions: [{catalog_key: "it-2565", plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]}]},
  {program_code: "DSBA", plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]},
];
const target = {dimension: "plan", program: "DSBA", operand: "right"};
const context = {program: "IT", catalog_key: "it-2565", plan: "coop", years: [2], semesters: [2], result_courses: [{program: "IT", catalog_key: "it-2565", course_code: "06016405"}]};
const session = {id: "a", program: "IT", catalogKey: "it-2565", plan: "coop", context, messages: [], title: "New chat", pendingClarification: null};
const entry = {id: 42, question: "original comparison", answer: "Any human-readable wording", planClarification: true, status: "clarification_required"};
function pending(clarificationTarget = target) {
  return applyChatResponse([structuredClone(session)], session.id, entry, {
    next_context: context, clarification_target: clarificationTarget,
  })[0];
}

test("normal plan clarification keeps normal selection and reset behavior", () => {
  const before = pending(null);
  const retry = selectPlanForRetry(before, "no_coop", programs);
  assert.equal(retry.session.plan, "no_coop");
  assert.equal(retry.session.program, "IT");
  assert.equal(retry.session.pendingClarification, null);
  assert.equal(retry.clarificationResolution, null);
  assert.deepEqual(buildConversationContext(retry.session), {program: "IT", catalog_key: "it-2565", plan: "no_coop"});
});

test("right operand retry preserves IT scope and sends one-shot resolution", async () => {
  const before = pending();
  assert.deepEqual(before.pendingClarification.clarification_target, target);
  const snapshot = structuredClone(before);
  const retry = selectPlanForRetry(before, "no_coop", programs);
  assert.deepEqual(before, snapshot);
  assert.equal(retry.session.plan, "coop");
  assert.equal(retry.session.program, "IT");
  assert.deepEqual(retry.session.context, context);
  assert.deepEqual(retry.clarificationResolution, {...target, value: "no_coop"});
  const requests = [];
  const oldFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    requests.push(JSON.parse(options.body));
    return {ok: true, json: async () => ({status: "answer", answer: "verified", next_context: context})};
  };
  try {
    const result = await askQuestion(entry.question, buildConversationContext(retry.session), retry.session.program, retry.clarificationResolution);
    assert.equal(result.status, "answer");
    await askQuestion("next normal IT query", buildConversationContext(retry.session), retry.session.program);
  } finally { globalThis.fetch = oldFetch; }
  assert.deepEqual(requests[0].clarification_resolution, {...target, value: "no_coop"});
  assert.equal(requests[0].home_program, "IT");
  assert.equal(requests[0].question, entry.question);
  assert.equal(requests[0].conversation_context.plan, "coop");
  assert.equal(requests[1].conversation_context.plan, "coop");
  assert.ok(!Object.hasOwn(requests[1], "clarification_resolution"));
  assert.ok(!Object.hasOwn(retry.session, "clarificationResolution"));
});

test("left operand metadata targets left without changing normal plan", () => {
  const left = {dimension: "plan", program: "IT", operand: "left"};
  const retry = selectPlanForRetry(pending(left), "no_coop", programs);
  assert.deepEqual(retry.clarificationResolution, {...left, value: "no_coop"});
  assert.equal(retry.session.plan, "coop");
});

test("operand choices come from target program rather than normal session", () => {
  const metadata = [...programs, {program_code: "AIT", plans: [{plan_key: "coop"}]}];
  const state = pending({dimension: "plan", program: "AIT", operand: "right"});
  assert.deepEqual(clarificationPlans(state, metadata), [{plan_key: "coop"}]);
  assert.throws(() => selectPlanForRetry(state, "no_coop", metadata));
});

test("malformed or unknown targets fail closed without scope mutation", () => {
  for (const value of [[], "right", {}, {...target, operand: "middle"}, {...target, dimension: "catalog"}, {...target, program: "UNKNOWN"}, {...target, extra: true}]) {
    const state = pending();
    const snapshot = structuredClone(state);
    assert.throws(() => selectPlanForRetry(state, "no_coop", programs,
      {action: "plan_required", clarification_target: value}));
    assert.deepEqual(state, snapshot);
  }
});

test("success replaces original message, clears pending, and isolates origin session", () => {
  const before = pending();
  const retry = selectPlanForRetry(before, "no_coop", programs);
  const other = {...structuredClone(session), id: "b", program: "DSBA", plan: "no_coop"};
  const answer = {...entry, answer: "verified", status: "answer", planClarification: false};
  const result = applyChatResponse([retry.session, other], "a", answer, {next_context: context}, entry.id);
  assert.deepEqual(result[0].messages, [answer]);
  assert.equal(result[0].pendingClarification, null);
  assert.equal(result[0].plan, "coop");
  assert.deepEqual(buildConversationContext(result[0]), context);
  assert.deepEqual(result[1], other);
  assert.ok(!Object.hasOwn(result[0], "clarification_resolution"));
});

test("page uses distinct sidebar and clarification selection paths", () => {
  const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");
  assert.match(page, /selectPlanForRetry\(selectionSession, plan, programs, clarification\)/);
  assert.match(page, /onChange=\{plan => handlePlanChange\(plan, m\.id\)\}/);
  assert.match(page, /askQuestion\(q, seed, homeProgram, clarificationResolution, clarificationResolutions\)/);
  assert.match(page, /plans=\{messagePlans\}/);
});

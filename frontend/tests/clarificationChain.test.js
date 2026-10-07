import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {askQuestion} from "../src/api.js";
import {applyChatResponse, buildConversationContext, newSession, selectCatalogForRetry,
  selectPlanForRetry, selectProgram} from "../src/chatScope.js";

const programs = ["DSBA", "IT"].map(program => ({program_code: program,
  editions: [2560, 2565].map(year => ({catalog_key: `${program.toLowerCase()}-${year}`, academic_year: String(year),
    plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]}))}));
const original = {id: "a", program: "", catalogKey: "", plan: null, context: null, messages: [], title: "New chat", pendingClarification: null};
const entry = {id: 42, question: "original comparison", status: "clarification_required", catalogClarification: true, planClarification: false};
const left = {dimension: "catalog", program: "DSBA", operand: "left", value: "dsba-2565"};
const right = {dimension: "catalog", program: "IT", operand: "right", value: "it-2565"};
const target = ({value, ...rest}) => rest;
function clarify(session, resolution, history = null, retryId = null) {
  const message = {...entry, catalogClarification: resolution.dimension === "catalog", planClarification: resolution.dimension === "plan"};
  return applyChatResponse([session], "a", message,
    {status: "clarification_required", clarification_target: target(resolution), next_context: null}, retryId, history)[0];
}

test("two catalog choices accumulate on the same message and clear on success", async () => {
  const pending = clarify(original, left);
  const first = selectCatalogForRetry(pending, "dsba-2565", programs);
  const next = clarify(first.session, right, first.clarificationResolutions, 42);
  assert.deepEqual(next.pendingClarification.clarification_resolutions, [left]);
  assert.deepEqual(next.pendingClarification.clarification_target, target(right));
  const second = selectCatalogForRetry(next, "it-2565", programs);
  assert.deepEqual(second.clarificationResolutions, [left, right]);
  const requests = [];const oldFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    requests.push(JSON.parse(options.body));
    return {ok: true, json: async () => ({status: "answer", next_context: null})};
  };
  try {
    await askQuestion(entry.question, null, null, null, first.clarificationResolutions);
    await askQuestion(entry.question, null, null, null, second.clarificationResolutions);
    await askQuestion("new normal question", null, null);
  } finally {globalThis.fetch = oldFetch;}
  assert.deepEqual(requests[0].clarification_resolutions, [left]);
  assert.deepEqual(requests[1].clarification_resolutions, [left, right]);
  assert.ok(!Object.hasOwn(requests[1], "clarification_resolution"));
  assert.ok(!Object.hasOwn(requests[2], "clarification_resolutions"));
  const answer = {...entry, status: "answer", catalogClarification: false};
  const other = {...original, id: "b"};
  const [done, untouched] = applyChatResponse([second.session, other], "a", answer,
    {status: "answer", next_context: null}, 42, second.clarificationResolutions);
  assert.deepEqual(done.messages, [answer]);
  assert.equal(done.pendingClarification, null);
  assert.equal(done.program, "");assert.equal(done.catalogKey, "");
  assert.equal(buildConversationContext(done), null);
  assert.deepEqual(untouched, other);
  assert.ok(!Object.hasOwn(done, "clarification_resolutions"));
});

test("catalog then plan clarification preserves previous dimensions", () => {
  let session = clarify(original, left);
  const first = selectCatalogForRetry(session, "dsba-2565", programs);
  session = clarify(first.session, right, first.clarificationResolutions, 42);
  const second = selectCatalogForRetry(session, "it-2565", programs);
  const plan = {dimension: "plan", program: "DSBA", operand: "left", value: "coop"};
  session = clarify(second.session, plan, second.clarificationResolutions, 42);
  const third = selectPlanForRetry(session, "coop", programs);
  assert.deepEqual(third.clarificationResolutions, [left, right, plan]);
  assert.equal(third.session.plan, null);
  assert.equal(third.session.context, null);
});

test("new question and explicit scope switch abandon prior chain", () => {
  const first = selectCatalogForRetry(clarify(original, left), "dsba-2565", programs);
  const pending = clarify(first.session, right, first.clarificationResolutions, 42);
  const newEntry = {...entry, id: 43, question: "different question"};
  const [newPending] = applyChatResponse([pending], "a", newEntry,
    {status: "clarification_required", clarification_target: target(right), next_context: null});
  assert.ok(!Object.hasOwn(newPending.pendingClarification, "clarification_resolutions"));
  assert.equal(selectProgram(pending, "IT", programs).pendingClarification, null);
  assert.equal(newSession("").pendingClarification, null);
  assert.ok(!Object.hasOwn(original, "clarification_resolutions"));
});

test("conflicting or malformed stored resolutions fail closed", () => {
  for (const history of [[left, {...left, value: "dsba-2560"}], "invalid", [null], [{...left, value: 2565}], [left]]) {
    const state = clarify(original, right);
    state.pendingClarification.clarification_resolutions = history;
    const before = structuredClone(state);
    if (Array.isArray(history) && history.length === 1 && history[0] === left) {
      assert.deepEqual(selectCatalogForRetry(state, "it-2565", programs).clarificationResolutions, [left, right]);
    } else {
      assert.throws(() => selectCatalogForRetry(state, "it-2565", programs));
    }
    assert.deepEqual(state, before);
  }
});

test("failed same-question retry retains chain without leaking it to other chat", () => {
  const first = selectCatalogForRetry(clarify(original, left), "dsba-2565", programs);
  const error = {...entry, status: "insufficient_evidence", catalogClarification: false};
  const other = {...original, id: "b"};
  const [pending, untouched] = applyChatResponse([first.session, other], "a", error,
    {status: "insufficient_evidence", next_context: null}, 42, first.clarificationResolutions);
  assert.deepEqual(pending.pendingClarification.clarification_resolutions, [left]);
  assert.deepEqual(untouched, other);
  assert.equal(pending.context, null);
});

test("page transports accumulated resolutions and binds them to response message", () => {
  const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");
  assert.match(page, /retry\.clarificationResolutions \? null : retry\.clarificationResolution, retry\.clarificationResolutions/);
  assert.match(page, /askQuestion\(q, seed, session\.program \|\| null, clarificationResolution, clarificationResolutions\)/);
  assert.match(page, /applyChatResponse\(previous, session\.id, entry, data, retryId,\s*clarificationResolutions/);
});

import assert from "node:assert/strict";
import test from "node:test";
import {askQuestion} from "../src/api.js";
import {applyChatResponse, buildConversationContext, clarificationCatalogs, selectCatalogForRetry} from "../src/chatScope.js";

const programs = [
  {program_code: "DSBA", editions: [
    {catalog_key: "dsba-2560", academic_year: "2560", plans: [{plan_key: "no_coop"}]},
    {catalog_key: "dsba-2565", academic_year: "2565", plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]},
  ]},
  {program_code: "IT", editions: [{catalog_key: "it-2565", academic_year: "2565", plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]}]},
];
const target = {dimension: "catalog", program: "DSBA", operand: null};
const question = "อยากรู้ว่าถ้าถอนวิชา Calculus 1 ในหลักสูตร DSBA จะส่งผลอย่างไร";
const session = {id: "a", program: "", catalogKey: "", plan: null, context: null, messages: [], title: "New chat"};
const entry = {id: 42, question, catalogClarification: true, status: "clarification_required", answer: "arbitrary human text"};
function pending(state = session) {
  return applyChatResponse([state], "a", entry, {status: "clarification_required", clarification_target: target, next_context: null})[0];
}

test("normal target supplies DSBA editions in an unscoped chat without parsing text", () => {
  const before = pending();
  assert.equal(before.program, "");
  assert.deepEqual(before.pendingClarification.clarification_target, target);
  assert.deepEqual(clarificationCatalogs(before, programs), programs[0].editions);
});

test("unscoped normal selection seeds DSBA context and retries original text without operand transport", async () => {
  const before = pending();const snapshot = structuredClone(before);
  const retry = selectCatalogForRetry(before, "dsba-2565", programs);
  assert.deepEqual(before, snapshot);
  assert.equal(retry.session.program, "");
  assert.equal(retry.session.catalogKey, "");
  assert.deepEqual(buildConversationContext(retry.session), {program: "DSBA", catalog_key: "dsba-2565"});
  assert.equal(retry.clarificationResolution, null);
  assert.equal(retry.clarificationResolutions, undefined);
  const requests = [];const oldFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    requests.push(JSON.parse(options.body));
    return {ok: true, json: async () => ({status: "answer", next_context: retry.session.context})};
  };
  try {
    await askQuestion(before.pendingClarification.question, buildConversationContext(retry.session), retry.session.program || null);
  } finally {globalThis.fetch = oldFetch;}
  assert.equal(requests[0].question, question);
  assert.equal(requests[0].home_program, null);
  assert.equal(requests[0].conversation_context.catalog_key, "dsba-2565");
  assert.ok(!Object.hasOwn(requests[0], "clarification_resolution"));
  assert.ok(!Object.hasOwn(requests[0], "clarification_resolutions"));
});

test("pinned DSBA selection updates normal catalog and preserves compatible plan", () => {
  const before = pending({...session, program: "DSBA", catalogKey: "dsba-2560", plan: "no_coop"});
  const retry = selectCatalogForRetry(before, "dsba-2565", programs);
  assert.equal(retry.session.program, "DSBA");
  assert.equal(retry.session.catalogKey, "dsba-2565");
  assert.equal(retry.session.plan, "no_coop");
  assert.equal(retry.session.pendingClarification, null);
});

test("wrong-program selection and pinned foreign target fail without mutation", () => {
  const before = pending();const snapshot = structuredClone(before);
  assert.throws(() => selectCatalogForRetry(before, "it-2565", programs));
  assert.deepEqual(before, snapshot);
  const pinned = pending({...session, program: "IT", catalogKey: "it-2565", plan: "coop"});
  assert.throws(() => selectCatalogForRetry(pinned, "dsba-2565", programs));
  assert.equal(pinned.program, "IT");
  assert.equal(pinned.catalogKey, "it-2565");
});

test("success replaces same pending message and isolates originating session", () => {
  const retry = selectCatalogForRetry(pending(), "dsba-2565", programs);
  const other = {...session, id: "b"};
  const answer = {...entry, status: "answer", catalogClarification: false, answer: "verified"};
  const [done, untouched] = applyChatResponse([retry.session, other], "a", answer,
    {status: "answer", next_context: retry.session.context}, 42);
  assert.deepEqual(done.messages, [answer]);
  assert.equal(done.pendingClarification, null);
  assert.equal(done.program, "");
  assert.equal(done.context.program, "DSBA");
  assert.equal(done.context.catalog_key, "dsba-2565");
  assert.deepEqual(untouched, other);
});

test("unscoped switch into DSBA clears old IT targets and plan-dependent context", () => {
  const before = pending({...session, context: {program: "IT", catalog_key: "it-2565", plan: "coop",
    years: [3], result_courses: [{course_code: "06016454", program: "IT", catalog_key: "it-2565"}]}});
  const retry = selectCatalogForRetry(before, "dsba-2565", programs);
  assert.deepEqual(retry.session.context, {program: "DSBA", catalog_key: "dsba-2565"});
  assert.equal(retry.session.program, "");
});

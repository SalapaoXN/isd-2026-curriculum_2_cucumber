import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {askQuestion} from "../src/api.js";
import {applyChatResponse, buildConversationContext, clarificationCatalogs, catalogOptionLabel,
  selectCatalog, selectCatalogForRetry, selectPlanForRetry} from "../src/chatScope.js";

const programs = [
  {program_code: "DSBA", editions: [
    {catalog_key: "dsba-2565", academic_year: "2565", plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]},
    {catalog_key: "dsba-2560", academic_year: "2560", plans: [{plan_key: "no_coop"}]},
  ]},
  {program_code: "IT", editions: [
    {catalog_key: "it-2560", academic_year: "2560"},
    {catalog_key: "it-2565", academic_year: "2565"},
  ]},
];
const context = {program: "DSBA", catalog_key: "dsba-2565", plan: "coop"};
const session = {id: "a", program: "DSBA", catalogKey: "dsba-2565", plan: "coop", context, messages: [], title: "New chat"};
const target = {dimension: "catalog", program: "IT", operand: "right"};
const entry = {id: 42, question: "original comparison", catalogClarification: true, status: "clarification_required"};
const pending = () => applyChatResponse([session], "a", entry, {next_context: context, clarification_target: target})[0];

test("catalog choices come from target program canonical metadata", () => {
  const editions = clarificationCatalogs(pending(), programs);
  assert.deepEqual(editions, programs[1].editions);
  assert.deepEqual(editions.map(e => catalogOptionLabel(e, editions)), ["2560", "2565"]);
  const duplicateYears = [{catalog_key: "edition-a", academic_year: "2565"}, {catalog_key: "edition-b", academic_year: "2565"}];
  assert.equal(catalogOptionLabel(duplicateYears[0], duplicateYears), "2565 (edition-a)");
});

test("operand catalog selection preserves home and sends one-shot retry only", async () => {
  const before = pending();const snapshot = structuredClone(before);
  const retry = selectCatalogForRetry(before, "it-2565", programs);
  assert.deepEqual(before, snapshot);
  assert.equal(retry.session.program, "DSBA");
  assert.equal(retry.session.catalogKey, "dsba-2565");
  assert.equal(retry.session.plan, "coop");
  assert.deepEqual(retry.session.context, context);
  assert.deepEqual(retry.clarificationResolution, {...target, value: "it-2565"});
  const requests = [];const oldFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    requests.push(JSON.parse(options.body));
    return {ok: true, json: async () => ({status: "answer", next_context: context})};
  };
  try {
    await askQuestion(entry.question, buildConversationContext(retry.session), "DSBA", retry.clarificationResolution);
    await askQuestion("normal DSBA query", buildConversationContext(retry.session), "DSBA");
  } finally {globalThis.fetch = oldFetch;}
  assert.deepEqual(requests[0].clarification_resolution, {...target, value: "it-2565"});
  assert.equal(requests[0].question, entry.question);
  assert.equal(requests[0].conversation_context.catalog_key, "dsba-2565");
  assert.ok(!Object.hasOwn(requests[1], "clarification_resolution"));
  assert.equal(requests[1].conversation_context.catalog_key, "dsba-2565");
});

test("normal catalog selection retains existing reset and compatible plan policy", () => {
  const selected = selectCatalog(session, "dsba-2560", programs);
  assert.equal(selected.catalogKey, "dsba-2560");
  assert.equal(selected.plan, "no_coop");
  assert.deepEqual(selected.context, {program: "DSBA", catalog_key: "dsba-2560", plan: "no_coop"});
  assert.equal(selected.pendingClarification, null);
  assert.equal(selectCatalogForRetry(session, "dsba-2560", programs).clarificationResolution, null);
});

test("wrong catalog or malformed target cannot mutate normal scope", () => {
  const before = pending();const snapshot = structuredClone(before);
  assert.throws(() => selectCatalogForRetry(before, "dsba-2565", programs));
  assert.deepEqual(before, snapshot);
  for (const value of [{...target, operand: "middle"}, {...target, dimension: "plan"}, {...target, program: "UNKNOWN"}, []]) {
    const state = {...before, pendingClarification: {...before.pendingClarification, clarification_target: value}};
    assert.throws(() => selectCatalogForRetry(state, "it-2565", programs));
  }
});

test("successful retry replaces original message and isolates originating session", () => {
  const retry = selectCatalogForRetry(pending(), "it-2565", programs);
  const other = {...session, id: "b", messages: []};
  const answer = {...entry, status: "answer", answer: "verified", catalogClarification: false};
  const result = applyChatResponse([retry.session, other], "a", answer, {next_context: context}, 42);
  assert.deepEqual(result[0].messages, [answer]);
  assert.equal(result[0].pendingClarification, null);
  assert.equal(result[0].catalogKey, "dsba-2565");
  assert.deepEqual(result[1], other);
  assert.ok(!Object.hasOwn(result[0], "clarification_resolution"));
});

test("plan popup keeps canonical plan retry separate from catalog selection", () => {
  const state = {...session, pendingClarification: {question: "compare", messageId: 42,
    clarification_target: {dimension: "plan", program: "DSBA", operand: "right"}}};
  const retry = selectPlanForRetry(state, "no_coop", programs);
  assert.equal(retry.session.plan, "coop");
  assert.equal(retry.clarificationResolution.value, "no_coop");
  const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");
  assert.match(page, /handleCatalogChange\(edition\.catalog_key, m\.id\)/);
  assert.match(page, /clarificationCatalogs\(clarificationSession, programs, clarification\)/);
  assert.match(page, /selectCatalogForRetry\(selectionSession, catalogKey, programs, clarification\)/);
  assert.match(page, /onChange=\{plan => handlePlanChange\(plan, m\.id\)\}/);
});

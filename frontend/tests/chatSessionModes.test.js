import assert from "node:assert/strict";
import test from "node:test";
import {askQuestion} from "../src/api.js";
import {
  applyChatResponse, buildConversationContext, newSession,
  normalizeStoredSession, selectProgram,
} from "../src/chatScope.js";

const programs = [
  {program_code: "IT", editions: [{catalog_key: "it-2565", academic_year: "2565", plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]}]},
  {program_code: "DSBA", editions: [{catalog_key: "dsba-2565", academic_year: "2565", plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]}]},
];

test("scoped request sends selected home and normal follow-up context", async t => {
  let payload;
  t.mock.method(globalThis, "fetch", async (_url, options) => {
    payload = JSON.parse(options.body);
    return {ok: true, json: async () => ({status: "answer"})};
  });
  const session = {...newSession("IT", "it-2565", programs), plan: "coop",
    context: {program: "IT", years: [2], semesters: [1], result_courses: []}};
  await askQuestion("แล้วเทอม 2 ล่ะ", buildConversationContext(session), session.program || null);
  assert.equal(payload.home_program, "IT");
  assert.deepEqual(payload.conversation_context, {program: "IT", years: [2], semesters: [1],
    result_courses: [], catalog_key: "it-2565", plan: "coop"});
});

test("unscoped request sends null even with a normal IT referent", async t => {
  let payload;
  t.mock.method(globalThis, "fetch", async (_url, options) => {
    payload = JSON.parse(options.body);
    return {ok: true, json: async () => ({status: "answer"})};
  });
  const session = {...newSession(""), context: {program: "IT", catalog_key: "it-2565"}};
  await askQuestion("DSBA ปี 2", buildConversationContext(session), session.program || null);
  assert.equal(payload.home_program, null);
  assert.equal(payload.conversation_context.program, "IT");
});

test("creating a chat from explicit unscoped selection preserves empty home", () => {
  const active = newSession("");
  const fresh = newSession(active.program ?? "IT", "", programs);
  assert.equal(fresh.program, "");
  assert.equal(fresh.catalogKey, "");
  assert.equal(fresh.plan, null);
  assert.equal(buildConversationContext(fresh), null);
});

test("explicit program selector resets dependent state and preserves messages", () => {
  const messages = [{id: 1, question: "old"}];
  const session = {...newSession("IT", "it-2565", programs), messages,
    plan: "coop", context: {program: "IT", years: [2], focus_course: {}, result_courses: [{}]},
    pendingClarification: {question: "pending"}};
  const selected = selectProgram(session, "DSBA", programs);
  assert.equal(selected.program, "DSBA");
  assert.equal(selected.catalogKey, "dsba-2565");
  assert.equal(selected.plan, null);
  assert.deepEqual(selected.context, {program: "DSBA", catalog_key: "dsba-2565"});
  assert.equal(selected.pendingClarification, null);
  assert.equal(selected.messages, messages);
});

test("comparison or foreign response cannot rewrite selected home fields", () => {
  const session = {...newSession("IT", "it-2565", programs), plan: "coop"};
  const response = {next_context: {program: "DSBA", catalog_key: "dsba-2565", plan: "no_coop",
    result_courses: [{course_code: "foreign"}]}, comparison: {right: {program: "DSBA"}}};
  const [updated] = applyChatResponse([session], session.id,
    {id: 1, question: "compare", planClarification: false}, response);
  assert.equal(updated.program, "IT");
  assert.equal(updated.catalogKey, "it-2565");
  assert.equal(updated.plan, "coop");
  assert.deepEqual(updated.context, {program: "IT", catalog_key: "it-2565", plan: "coop"});
});

test("late response updates originating chat only", async () => {
  const a = {...newSession("IT", "it-2565", programs), id: "A"};
  const b = {...newSession("DSBA", "dsba-2565", programs), id: "B"};
  let finish;
  const pending = new Promise(resolve => {finish = resolve;});
  const originId = a.id;
  let sessions = [a, b];
  const delivery = pending.then(data => {
    sessions = applyChatResponse(sessions, originId,
      {id: 1, question: "from A", planClarification: false}, data);
  });
  // User switches to B while the originating request is outstanding.
  const activeId = b.id;
  finish({next_context: {program: "IT", catalog_key: "it-2565", years: [2]}});
  await delivery;
  assert.equal(activeId, "B");
  assert.equal(sessions[1], b);
  assert.equal(sessions[0].messages.length, 1);
  assert.deepEqual(sessions[0].context.years, [2]);
});

test("stored sessions reuse selected program without inference from history", () => {
  for (const program of ["IT", ""]) {
    const session = normalizeStoredSession({id: "old", program,
      context: {program: "DSBA"}, messages: []});
    assert.equal(session.program, program);
  }
  for (const program of [undefined, {}, 7]) {
    const session = normalizeStoredSession({id: "old", program,
      catalogKey: "dsba-2565", plan: "coop", context: {program: "DSBA"}, messages: null});
    assert.equal(session.program, "");
    assert.equal(session.context, null);
    assert.equal(session.catalogKey, "");
    assert.equal(session.plan, null);
    assert.deepEqual(session.messages, []);
  }
  assert.equal(normalizeStoredSession(null), null);
  assert.equal(normalizeStoredSession({}), null);
});

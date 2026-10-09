import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";

import {
  applyChatResponse,
  normalizeStoredSession,
  selectProgramForRetry,
  updateMessageClarification,
} from "../src/chatScope.js";

const programs = [
  {program_code: "IT", editions: [
    {catalog_key: "it-2565", academic_year: "2565", plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]},
  ]},
  {program_code: "DSBA", editions: [
    {catalog_key: "dsba-2565", academic_year: "2565", plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]},
  ]},
];

const originalScope = {home_program: null, conversation_context: null};

function initialSession() {
  return {id: "chat", program: "", catalogKey: "", plan: null,
    context: null, messages: [], title: "New chat"};
}

function clarificationReply(id, question, action, requestScope = originalScope) {
  return {
    entry: {
      id,
      question,
      answer: "กรุณาระบุหลักสูตร",
      status: "clarification_required",
      programClarification: action === "program_required",
      catalogClarification: action === "catalog_required",
      planClarification: action === "plan_required",
      request_scope: requestScope,
    },
    data: {
      status: "clarification_required",
      action,
      clarification_target: null,
      next_context: null,
    },
  };
}

function normalReply(id, question) {
  return {
    entry: {id, question, answer: `answer:${question}`, status: "answer",
      programClarification: false, catalogClarification: false, planClarification: false},
    data: {status: "answer", next_context: null},
  };
}

test("program_required is retained on its originating message", () => {
  const session = initialSession();
  const reply = clarificationReply(1, "original course question", "program_required");
  const [updated] = applyChatResponse([session], session.id, reply.entry, reply.data);

  assert.equal(updated.messages[0].clarification.action, "program_required");
  assert.deepEqual(updated.messages[0].clarification.request_scope, originalScope);
  assert.equal(updated.pendingClarification.messageId, 1);
});

test("program options render from canonical metadata on the clarifying message", () => {
  const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");
  assert.match(page, /clarification\?\.action === "program_required"/);
  assert.match(page, /programs\.map\(program/);
  assert.match(page, /handleProgramClarificationChange\(program\.program_code, m\.id\)/);
});

for (const program of ["IT", "DSBA"]) {
  test(`selecting ${program} retries the original question with ${program} scope`, () => {
    const session = initialSession();
    const reply = clarificationReply(1, "original course question", "program_required");
    const [pending] = applyChatResponse([session], session.id, reply.entry, reply.data);
    const retry = selectProgramForRetry(
      pending,
      program,
      programs,
      pending.messages[0].clarification,
    );
    const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");

    assert.equal(retry.session.program, program);
    assert.equal(retry.retryScope.home_program, program);
    assert.equal(retry.retryScope.conversation_context.program, program);
    assert.equal(retry.retryScope.conversation_context.catalog_key,
      program === "IT" ? "it-2565" : "dsba-2565");
    assert.equal(retry.clarification.request_scope.home_program, program);
    assert.match(page,
      /handleProgramClarificationChange[\s\S]*?handleAsk\(\s*message\.question,\s*retry\.session,\s*message\.id/);
    assert.match(page, /retry\.retryScope/);
  });
}

test("a later unrelated response does not clear the earlier program clarification", () => {
  const session = initialSession();
  const q1 = clarificationReply(1, "Q1", "program_required");
  const [afterQ1] = applyChatResponse([session], session.id, q1.entry, q1.data);
  const q2 = normalReply(2, "Q2");
  const [afterQ2] = applyChatResponse([afterQ1], session.id, q2.entry, q2.data);

  assert.equal(afterQ2.messages.find(message => message.id === 1).clarification.action,
    "program_required");
  assert.equal(afterQ2.messages.find(message => message.id === 2).clarification, undefined);
  assert.equal(afterQ2.pendingClarification.messageId, 1);
});

test("program and catalog clarifications can coexist on separate messages", () => {
  const session = initialSession();
  const q1 = clarificationReply(1, "Q1", "program_required");
  const [afterQ1] = applyChatResponse([session], session.id, q1.entry, q1.data);
  const q2 = clarificationReply(2, "Q2", "catalog_required", {
    home_program: "IT", conversation_context: {program: "IT", catalog_key: null},
  });
  const [afterQ2] = applyChatResponse([afterQ1], session.id, q2.entry, q2.data);

  assert.equal(afterQ2.messages.find(message => message.id === 1).clarification.action,
    "program_required");
  assert.equal(afterQ2.messages.find(message => message.id === 2).clarification.action,
    "catalog_required");
});

test("retrying Q1 program choice leaves Q2 catalog clarification intact", () => {
  const session = initialSession();
  const q1 = clarificationReply(1, "Q1", "program_required");
  const [afterQ1] = applyChatResponse([session], session.id, q1.entry, q1.data);
  const q2 = clarificationReply(2, "Q2", "catalog_required", {
    home_program: "IT", conversation_context: {program: "IT", catalog_key: null},
  });
  const [afterQ2] = applyChatResponse([afterQ1], session.id, q2.entry, q2.data);
  const q1Clarification = afterQ2.messages.find(message => message.id === 1).clarification;
  const retry = selectProgramForRetry(
    afterQ2,
    "DSBA",
    programs,
    q1Clarification,
  );
  const [retried] = updateMessageClarification(
    [afterQ2], session.id, 1,
    retry.clarification,
  );

  assert.equal(retried.messages.find(message => message.id === 1).clarification.action,
    "program_required");
  assert.equal(retried.messages.find(message => message.id === 2).clarification.action,
    "catalog_required");
  const q1Answer = normalReply(1, "Q1");
  const [afterRetry] = applyChatResponse([retried], session.id, q1Answer.entry,
    q1Answer.data, 1);
  assert.equal(afterRetry.messages.find(message => message.id === 1).clarification, undefined);
  assert.equal(afterRetry.messages.find(message => message.id === 2).clarification.action,
    "catalog_required");
});

test("out-of-order catalog and program responses remain attached to their messages", () => {
  const session = initialSession();
  const q2 = clarificationReply(2, "Q2", "catalog_required", {
    home_program: "IT", conversation_context: {program: "IT", catalog_key: null},
  });
  const [afterQ2] = applyChatResponse([session], session.id, q2.entry, q2.data);
  const q1 = clarificationReply(1, "Q1", "program_required");
  const [afterQ1] = applyChatResponse([afterQ2], session.id, q1.entry, q1.data);

  assert.deepEqual(afterQ1.messages.map(message => [message.id, message.clarification?.action]), [
    [2, "catalog_required"], [1, "program_required"],
  ]);
});

test("program clarification persists through bounded session normalization", () => {
  const reply = clarificationReply(42, "original course question", "program_required");
  const [session] = applyChatResponse(
    [initialSession()], "chat", reply.entry, reply.data,
  );
  const stored = normalizeStoredSession(JSON.parse(JSON.stringify(session)));

  assert.equal(stored.messages[0].clarification.action, "program_required");
  assert.deepEqual(stored.messages[0].clarification.request_scope, originalScope);
  assert.equal(stored.pendingClarification.messageId, 42);
});

test("existing catalog and plan message clarification normalization is preserved", () => {
  const session = initialSession();
  const catalog = clarificationReply(1, "catalog question", "catalog_required", {
    home_program: null, conversation_context: null,
  });
  const [afterCatalog] = applyChatResponse([session], session.id, catalog.entry, catalog.data);
  const plan = clarificationReply(2, "plan question", "plan_required", {
    home_program: "IT", conversation_context: {program: "IT", catalog_key: "it-2565"},
  });
  const [afterPlan] = applyChatResponse([afterCatalog], session.id, plan.entry, plan.data);
  const restored = normalizeStoredSession(JSON.parse(JSON.stringify(afterPlan)));

  assert.equal(restored.messages.find(message => message.id === 1).clarification.action,
    "catalog_required");
  assert.equal(restored.messages.find(message => message.id === 2).clarification.action,
    "plan_required");
});

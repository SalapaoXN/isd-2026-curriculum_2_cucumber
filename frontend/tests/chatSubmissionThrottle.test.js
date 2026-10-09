import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";

import {applyChatResponse} from "../src/chatScope.js";
import {CHAT_SUBMISSION_INTERVAL_MS, createChatSubmissionManager} from "../src/chatSubmission.js";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return {promise, resolve, reject};
}

function enableClock(t) {
  t.mock.timers.enable({apis: ["Date"], now: 0});
}

function simulateAttempt(manager, sessionId, question, state) {
  const submission = manager.begin(sessionId, requestId => {
    state.apiCalls.push({requestId, question});
    state.acceptedBubbles.push(question);
    state.draft = "";
    return new Promise(() => {});
  });
  return submission;
}

test("first question is accepted and invokes the send operation", t => {
  enableClock(t);
  const manager = createChatSubmissionManager();
  const state = {apiCalls: [], acceptedBubbles: [], draft: "first question"};

  const submission = simulateAttempt(manager, "session-a", state.draft, state);

  assert.equal(submission.accepted, true);
  assert.equal(state.apiCalls.length, 1);
  assert.deepEqual(state.acceptedBubbles, ["first question"]);
  assert.equal(state.draft, "");
  assert.equal(manager.remaining("session-a"), CHAT_SUBMISSION_INTERVAL_MS);
});

test("Send-button attempt during cooldown does not call API, append bubble, or clear input", t => {
  enableClock(t);
  const manager = createChatSubmissionManager();
  const state = {apiCalls: [], acceptedBubbles: [], draft: "first question"};
  simulateAttempt(manager, "session-a", state.draft, state);
  state.draft = "second question";

  const blocked = simulateAttempt(manager, "session-a", state.draft, state);

  assert.equal(blocked.accepted, false);
  assert.equal(state.apiCalls.length, 1);
  assert.deepEqual(state.acceptedBubbles, ["first question"]);
  assert.equal(state.draft, "second question");
});

test("Ctrl+Enter attempt during cooldown uses the same gate and preserves input", t => {
  enableClock(t);
  const manager = createChatSubmissionManager();
  const state = {apiCalls: [], acceptedBubbles: [], draft: "first question"};
  simulateAttempt(manager, "session-a", state.draft, state);
  state.draft = "second question";

  // Ctrl+Enter and Send both reach the same handleAsk submission gate.
  const blocked = simulateAttempt(manager, "session-a", state.draft, state);

  assert.equal(blocked.accepted, false);
  assert.equal(state.apiCalls.length, 1);
  assert.equal(state.draft, "second question");
});

test("Q2 is accepted at 3000 ms while Q1 is unresolved", t => {
  enableClock(t);
  const manager = createChatSubmissionManager();
  const q1 = deferred();
  const q2 = deferred();
  const first = manager.begin("session-a", () => q1.promise);
  t.mock.timers.tick(CHAT_SUBMISSION_INTERVAL_MS);
  const second = manager.begin("session-a", () => q2.promise);

  assert.equal(first.accepted, true);
  assert.equal(second.accepted, true);
  assert.notEqual(first.requestId, second.requestId);
  assert.equal(manager.remaining("session-a"), CHAT_SUBMISSION_INTERVAL_MS);
});

test("Q3 is accepted after another interval with Q1 and Q2 still unresolved", t => {
  enableClock(t);
  const manager = createChatSubmissionManager();
  const pending = [];
  for (let index = 1; index <= 3; index += 1) {
    if (index > 1) t.mock.timers.tick(CHAT_SUBMISSION_INTERVAL_MS);
    pending.push(manager.begin("session-a", () => new Promise(() => {})));
  }

  assert.deepEqual(pending.map(item => item.accepted), [true, true, true]);
  assert.equal(new Set(pending.map(item => item.requestId)).size, 3);
});

test("submission count is unlimited when each question respects the interval", t => {
  enableClock(t);
  const manager = createChatSubmissionManager();
  let calls = 0;

  for (let index = 0; index < 25; index += 1) {
    if (index > 0) t.mock.timers.tick(CHAT_SUBMISSION_INTERVAL_MS);
    const submission = manager.begin("session-a", () => {
      calls += 1;
      return new Promise(() => {});
    });
    assert.equal(submission.accepted, true);
  }

  assert.equal(calls, 25);
});

test("out-of-order completions retain each submitted question and response", async t => {
  enableClock(t);
  const manager = createChatSubmissionManager();
  const session = {
    id: "session-a", program: "DSBA", catalogKey: "dsba-2565", plan: "no_coop",
    context: null, messages: [], title: "test",
  };
  let sessions = [session];
  const q1 = deferred();
  const first = manager.begin("session-a", () => q1.promise);
  t.mock.timers.tick(CHAT_SUBMISSION_INTERVAL_MS);
  const q2 = deferred();
  const second = manager.begin("session-a", () => q2.promise);

  const apply = (submission, question, promise) => promise.then(data => {
    const entry = {
      id: submission.requestId,
      question,
      answer: data.answer,
      status: data.status,
      planClarification: false,
      catalogClarification: false,
      provenance: [],
    };
    sessions = applyChatResponse(sessions, session.id, entry, data);
  });
  const firstResponse = apply(first, "Q1", q1.promise);
  const secondResponse = apply(second, "Q2", q2.promise);
  q2.resolve({status: "answer", answer: "answer to Q2", next_context: null});
  await secondResponse;
  q1.resolve({status: "answer", answer: "answer to Q1", next_context: null});
  await firstResponse;

  const messages = sessions[0].messages;
  assert.equal(messages.length, 2);
  assert.deepEqual(
    Object.fromEntries(messages.map(message => [message.question, message.answer])),
    {Q2: "answer to Q2", Q1: "answer to Q1"},
  );
  assert.equal(new Set(messages.map(message => message.id)).size, 2);
});

test("normal one-question flow still reaches the shared handler from Send and Ctrl+Enter", () => {
  const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");
  const handler = page.match(/async function handleAsk\([\s\S]*?(?=\n  if \(!active\))/)[0];
  assert.match(page, /onClick=\{handleAsk\}/);
  assert.match(page, /\(e\.ctrlKey \|\| e\.metaKey\) && e\.key === "Enter"\) handleAsk\(\)/);
  assert.match(page, /submissionManagerRef\.current\.begin\([\s\S]{0,80}session\.id/);
  assert.match(page, /disabled=\{cooldownRemainingMs > 0\}/);
  assert.doesNotMatch(handler, /if\s*\(loading\)\s*return/);
  assert.ok(handler.indexOf("if (!submission.accepted) return") < handler.indexOf("setQuestion(current"));
  assert.match(page, /setTimeout\(\(\) => \{[\s\S]*?setCooldownNow\(Date\.now\(\)\)[\s\S]*?submission\.availableAt - Date\.now\(\)/);
  assert.match(page, /id: retryId \|\| submission\.requestId/);
});

test("clarification retries bypass the question throttle and retain the original question", () => {
  const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");
  assert.match(page, /submissionManagerRef\.current\.begin\([\s\S]{0,500}bypassCooldown:\s*Boolean\(retryId\)/);
  assert.match(page, /handleAsk\(message\.question, retry\.session, message\.id/);
});

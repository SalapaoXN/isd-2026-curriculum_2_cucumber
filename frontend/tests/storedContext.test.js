import test from "node:test";
import assert from "node:assert/strict";

import {
  normalizeStoredContext,
  normalizeStoredSession,
} from "../src/chatScope.js";

test("stored context keeps bounded known fields and drops abuse", () => {
  const cleaned = normalizeStoredContext({
    program: "IT",
    catalog_key: "it-2565",
    plan: "coop",
    years: [2],
    semesters: [1],
    focus_course: {course_code: "06016405", program: "IT", catalog_key: "it-2565"},
    result_courses: [{course_code: "06016405"}],
    result_scope_program: "IT",
    last_normal_operation: {
      kind: "list_courses", program: "IT", catalog_key: "it-2565",
      plan: "coop", years: [2], semesters: [1],
    },
  });
  assert.equal(cleaned.program, "IT");
  assert.equal(cleaned.result_courses.length, 1);
  assert.equal(cleaned.last_normal_operation.kind, "list_courses");
});

test("stored context discards forged and malformed state", () => {
  assert.equal(
    normalizeStoredContext({
      program: "IT",
      injected: {nested: "x"},
      left: {program: "DSBA"},
      years: [true, "2"],
      semesters: "x",
      result_courses: [{course_code: 42}, {nope: true}],
      last_normal_operation: {kind: "compare", program: "IT"},
    }).injected,
    undefined,
  );
  const cleaned = normalizeStoredContext({
    program: "IT",
    left: {program: "DSBA"},
    years: [true, "2"],
    result_courses: new Array(25).fill({course_code: "06016405"}),
  });
  assert.ok(!("left" in cleaned));
  assert.deepEqual(cleaned.years, []);
  assert.equal(cleaned.result_courses.length, 25);
  assert.equal(normalizeStoredContext({injected: 1}), null);
  assert.equal(normalizeStoredContext("nope"), null);
});

test("stored sessions normalize malformed context instead of keeping it", () => {
  const session = normalizeStoredSession({
    id: "s1",
    program: "IT",
    context: {program: "IT", injected: {a: 1}, years: [true]},
  });
  assert.deepEqual(session.context, {program: "IT", years: []});
});

test("bounded normalization preserves reference objects, empty ownership and normal-operation keys", () => {
  const last = {route: "course", course_code: "06016414", operations: ["sum_credits"], program: "IT"};
  const cleaned = normalizeStoredContext({program: "IT", result_courses: [], result_set_empty: true,
    last_answer: last, years: [0, 1, 2, 3, 4, 5, 5, 5], semesters: [1, 2, 3, 2],
    last_normal_operation: {kind: "list_courses", program: "IT", catalog_key: null, plan: null, years: [1], semesters: []}});
  assert.deepEqual(cleaned.last_answer, last);
  assert.deepEqual(cleaned.result_courses, []);
  assert.equal(cleaned.years.length, 6);
  assert.equal(cleaned.semesters.length, 3);
  assert.deepEqual(Object.keys(cleaned.last_normal_operation).sort(), ["catalog_key", "kind", "plan", "program", "semesters", "years"]);
});

test("stored message-local clarification survives normalization with bounded fields", () => {
  const local = {
    action: "catalog_required",
    clarification_target: {dimension: "catalog", program: "DSBA", operand: null},
    next_context: {program: "DSBA", catalog_key: "dsba-2565", pending_catalog_selection: true, injected: "discard"},
    request_scope: {home_program: "DSBA", conversation_context: {program: "DSBA", catalog_key: "dsba-2565", plan: "coop", injected: true}, extra: "discard"},
    clarification_resolutions: [{dimension: "catalog", program: "IT", operand: "right", value: "it-2565"}],
    unbounded_extra: "discard",
  };
  const session = normalizeStoredSession({
    id: "local-state",
    program: "DSBA",
    messages: [{id: 42, question: "Q1", clarification: local}],
  });

  assert.deepEqual(session.messages[0].clarification, {
    action: "catalog_required",
    clarification_target: {dimension: "catalog", program: "DSBA", operand: null},
    next_context: {program: "DSBA", catalog_key: "dsba-2565", pending_catalog_selection: true},
    request_scope: {home_program: "DSBA", conversation_context: {program: "DSBA", catalog_key: "dsba-2565", plan: "coop"}},
    clarification_resolutions: [{dimension: "catalog", program: "IT", operand: "right", value: "it-2565"}],
  });
});

test("legacy pending clarification migrates to its matching message only", () => {
  const target = {dimension: "catalog", program: "DSBA", operand: null};
  const migrated = normalizeStoredSession({
    id: "legacy-matched",
    program: "DSBA",
    context: {program: "DSBA", catalog_key: "dsba-2565", plan: "coop"},
    messages: [{id: 42, question: "Q1", catalogClarification: true}],
    pendingClarification: {question: "Q1", messageId: 42, clarification_target: target},
  });
  assert.equal(migrated.messages[0].clarification.action, "catalog_required");
  assert.deepEqual(migrated.messages[0].clarification.clarification_target, target);
  assert.equal(migrated.messages[0].clarification.request_scope.home_program, "DSBA");
  assert.equal(migrated.pendingClarification.messageId, 42);

  const orphan = normalizeStoredSession({
    id: "legacy-orphan",
    program: "DSBA",
    messages: [{id: 7, question: "Q2"}],
    pendingClarification: {question: "Q1", messageId: 42, clarification_target: target},
  });
  assert.equal(orphan.messages[0].clarification, undefined);
  assert.equal(orphan.pendingClarification, null);
});

test("normalized message-local clarification overrides stale legacy pointer", () => {
  const localTarget = {dimension: "plan", program: "DSBA", operand: "right"};
  const staleTarget = {dimension: "catalog", program: "IT", operand: "left"};
  const session = normalizeStoredSession({
    id: "local-priority",
    program: "DSBA",
    messages: [{id: 42, question: "compare", clarification: {
      action: "plan_required", clarification_target: localTarget,
      request_scope: {home_program: "DSBA", conversation_context: {program: "DSBA", catalog_key: "dsba-2565"}},
    }}],
    pendingClarification: {question: "compare", messageId: 42, clarification_target: staleTarget},
  });
  assert.deepEqual(session.messages[0].clarification.clarification_target, localTarget);
  assert.deepEqual(session.pendingClarification.clarification_target, localTarget);
});

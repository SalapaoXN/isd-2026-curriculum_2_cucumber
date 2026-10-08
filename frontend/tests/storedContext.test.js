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

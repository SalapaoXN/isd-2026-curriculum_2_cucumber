import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

import {
  buildConversationContext,
  applyChatResponse,
  defaultCatalogKey,
  resetContextForEdition,
  availablePlans,
  displayablePlans,
  selectPlan,
  planLabel,
  formatElapsedTime,
  provenanceLabel,
  classifyAnswerLine,
} from "../src/chatScope.js";

test("provenance display hides internal filenames without changing metadata", () => {
  const source = {program: "IT", source_page: 328, source_filename: "it_page_328.png"};
  assert.equal(provenanceLabel(source), "หลักสูตร IT · หน้า 328");
  assert.equal(source.source_filename, "it_page_328.png");
  assert.equal(provenanceLabel({source_page: 359}), "หน้า 359");
  assert.equal(provenanceLabel({program: "DSBA"}), "หลักสูตร DSBA");
  assert.equal(provenanceLabel({source_filename: "internal.json"}), "");
});

test("elapsed time formatting is bounded and absent for old messages", () => {
  assert.equal(formatElapsedTime(820), "0.82 วินาที");
  assert.equal(formatElapsedTime(2400), "2.4 วินาที");
  assert.equal(formatElapsedTime(37900), "37.9 วินาที");
  assert.equal(formatElapsedTime(1000), "1.0 วินาที");
  assert.equal(formatElapsedTime(0), "0.00 วินาที");
  for (const value of [undefined, null, NaN, -1]) assert.equal(formatElapsedTime(value), null);
});

test("answer line presentation classifies only visible shapes without rewriting text", () => {
  assert.equal(classifyAnswerLine("ปี 1 เทอม 1:"), "section-heading");
  assert.equal(classifyAnswerLine("รายวิชาที่มีวิชาบังคับก่อนในหลักสูตร AIT:"), "section-heading");
  assert.equal(classifyAnswerLine("- 06026201 CALCULUS 2"), "list-item");
  assert.equal(classifyAnswerLine("• รายวิชา"), "list-item");
  assert.equal(classifyAnswerLine("  วิชาบังคับก่อน: 06026200"), "prerequisite");
  assert.equal(classifyAnswerLine("หมายเหตุ: แสดงข้อมูลที่ยืนยันได้"), "note");
  assert.equal(classifyAnswerLine("ช่องวิชาเลือกที่ยังไม่ระบุวิชาจริง"), "elective-slot");
  assert.equal(classifyAnswerLine("ช่องวิชาเลือก: เลือก 1 วิชา"), "elective-slot");
  assert.equal(classifyAnswerLine("อ้างอิง: หน้า 22"), "reference-note");
  assert.equal(classifyAnswerLine("06026201 ชื่อภาษาไทย: แคลคูลัส 2; CALCULUS 2"), "paragraph");
  assert.equal(classifyAnswerLine(""), "blank");
});

test("each request including plan retry records its own timing and keeps expandable provenance", () => {
  const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");
  assert.match(page, /performance\.now\(\);\s*const data = await askQuestion\(q, seed, session\.program \|\| null, clarificationResolution, clarificationResolutions\);\s*const elapsedMs = performance\.now\(\) - requestStarted/);
  assert.match(page, /elapsedMs,/);
  assert.match(page, /CUCUMBER\s*\{formatElapsedTime\(m\.elapsedMs\)/);
  assert.match(page, /handleAsk\(pending\.question, selected, pending\.messageId, clarificationResolution, clarificationResolutions\)/);
  assert.match(page, /<details className="chat-provenance">/);
  assert.match(page, /แหล่งอ้างอิง \{m\.provenance\.length\} รายการ/);
  assert.match(page, /provenanceLabel\(source\)/);
  assert.match(page, /<AnswerText text=\{/);
  assert.doesNotMatch(page, /\{source\.source_filename\}/);
});

test("plan selection uses edition options and resets targets without deleting history", () => {
  const metadata = [{program_code: "IT", editions: [{catalog_key: "it-2565", plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]}]}];
  const session = {program: "IT", catalogKey: "it-2565", messages: [{question: "old"}], context: {course_code: "06016414", result_courses: [{}]}, pendingClarification: {question: "pending"}};
  assert.deepEqual(availablePlans(session, metadata).map(p => p.plan_key), ["coop", "no_coop"]);
  for (const plan of ["coop", "no_coop"]) {
    const selected = selectPlan(session, plan, metadata);
    assert.equal(selected.plan, plan);
    assert.deepEqual(buildConversationContext(selected), {program: "IT", catalog_key: "it-2565", plan});
    assert.deepEqual(selected.messages, session.messages);
    assert.equal(selected.pendingClarification, null);
  }
  assert.equal(planLabel("coop"), "สหกิจ");
  assert.equal(planLabel("no_coop"), "ไม่สหกิจ");
  assert.equal(planLabel("default"), null);
  assert.equal(planLabel("mystery"), null);
  assert.deepEqual(
    displayablePlans([{plan_key: "coop"}, {plan_key: "no_coop"}]).map((p) => p.plan_key),
    ["coop", "no_coop"]
  );
  assert.deepEqual(displayablePlans([{plan_key: "default"}]), []);
  assert.throws(() => selectPlan({...session, catalogKey: "other"}, "coop", metadata));
});

test("internal default plan is hidden from plan control and scope label", () => {
  const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");
  assert.match(page, /visiblePlans = displayablePlans\(plans\)/);
  assert.match(page, /<PlanSelector plans=\{visiblePlans\}/);
  assert.match(page, /visiblePlans\.length > 0/);
  assert.match(page, /planSegment \? ` · \$\{planSegment\}` : ""/);
});

test("plan choices use accessible shared markup and structured pending-question retry", () => {
  const component = readFileSync(new URL("../src/components/PlanSelector.jsx", import.meta.url), "utf8");
  const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");
  assert.match(component, /<button/);
  assert.match(component, /aria-pressed=/);
  assert.match(component, /role="group"/);
  assert.equal((page.match(/<PlanSelector /g) || []).length, 2);
  assert.match(page, /data\.action === "plan_required"/);
  assert.match(page, /handleAsk\(pending\.question, selected, pending\.messageId, clarificationResolution, clarificationResolutions\)/);
  const session = {id: "chat-it", program: "IT", catalogKey: "it-2565", plan: null,
    title: "New chat", messages: [], context: null};
  const entry = {id: 42, question: "ปี 2 มีอะไรบ้าง", planClarification: true,
    status: "clarification_required", answer: "กรุณาระบุแผนการเรียน"};
  const [pending] = applyChatResponse([session], session.id, entry,
    {status: "clarification_required", action: "plan_required", next_context: null});
  assert.deepEqual(pending.pendingClarification, {question: entry.question, messageId: entry.id});
  assert.deepEqual(pending.messages, [entry]);
  const metadata = [{program_code: "IT", editions: [{catalog_key: "it-2565", plans: [{plan_key: "coop"}, {plan_key: "no_coop"}]}]}];
  const selected = selectPlan(pending, "coop", metadata);
  assert.equal(selected.pendingClarification, null);
  assert.deepEqual(selected.context, {program: "IT", catalog_key: "it-2565", plan: "coop"});
  const retry = {...entry, planClarification: false, status: "answer", answer: "verified"};
  const [answered] = applyChatResponse([selected], session.id, retry,
    {status: "answer", next_context: selected.context}, entry.id);
  assert.deepEqual(answered.messages, [retry]);
  assert.equal(answered.pendingClarification, null);
  assert.equal(answered.program, "IT");
  assert.equal(answered.plan, "coop");
  assert.doesNotMatch(page, /\b(?:alert|prompt|confirm)\s*\(/);
});

test("selected plan survives edition reset and no plan is forced for multi-plan scope", () => {
  assert.deepEqual(buildConversationContext({program: "IT", catalogKey: "it-2565", plan: "no_coop", context: {catalog_key: "it-2560", course_code: "old"}}),
    {program: "IT", catalog_key: "it-2565", plan: "no_coop"});
  assert.deepEqual(buildConversationContext({program: "IT", catalogKey: "it-2565", plan: null, context: null}),
    {program: "IT", catalog_key: "it-2565"});
});

const programs = [
  {
    program_code: "DSBA",
    editions: [
      { catalog_key: "dsba-2565", academic_year: "2565" },
      { catalog_key: "dsba-2560", academic_year: "2560" },
    ],
  },
  {
    program_code: "IT",
    editions: [{ catalog_key: "it-2564", academic_year: "2564" }],
  },
];

test("default edition uses the newest API-provided year and handles one edition", () => {
  assert.equal(defaultCatalogKey("DSBA", programs), "dsba-2565");
  assert.equal(defaultCatalogKey("IT", programs), "it-2564");
  assert.equal(
    defaultCatalogKey("UNKNOWN", [
      {
        program_code: "UNKNOWN",
        editions: [
          { catalog_key: "unknown-a", academic_year: "unknown" },
          { catalog_key: "unknown-b", academic_year: "unknown" },
        ],
      },
    ]),
    ""
  );
});

test("switching edition clears prior result, focus, and plan context", () => {
  const prior = {
    program: "DSBA",
    catalog_key: "dsba-2565",
    plan: "coop",
    focused_course_code: "06066300",
    result_courses: [{ catalog_key: "dsba-2565", course_code: "06066300" }],
  };

  assert.deepEqual(resetContextForEdition("DSBA", "dsba-2560", prior), {
    program: "DSBA",
    catalog_key: "dsba-2560",
  });
  assert.deepEqual(
    resetContextForEdition("DSBA", "dsba-2565", {
      ...prior,
      catalog_key: "dsba-2560",
      result_courses: [{ catalog_key: "dsba-2560", course_code: "06026106" }],
    }),
    { program: "DSBA", catalog_key: "dsba-2565" }
  );
});

test("selected session edition overrides a stale context before sending", () => {
  const context = buildConversationContext({
    program: "DSBA",
    catalogKey: "dsba-2560",
    context: {
      program: "DSBA",
      catalog_key: "dsba-2565",
      plan: "coop",
      result_courses: [{ catalog_key: "dsba-2565", course_code: "06066300" }],
    },
  });

  assert.deepEqual(context, { program: "DSBA", catalog_key: "dsba-2560" });
});

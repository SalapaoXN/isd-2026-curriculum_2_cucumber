import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";

// Exercise the actual AnswerText prop expression without a second copy of
// presentation logic or introducing a JSX test framework for this small fix.
const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");
const expression = page.match(/<AnswerText text=\{([\s\S]*?)\} \/>/)[1];
const displayedText = new Function("m", `return (${expression});`);

test("plan_required preserves backend operand/program clarification", () => {
  const answer = "กรุณาระบุแผนการเรียนของ DSBA ที่ต้องการใช้ในการเปรียบเทียบฝั่งขวา";
  assert.equal(displayedText({answer, planClarification: true}), answer);
});

test("catalog_required preserves specific catalog question", () => {
  const answer = "กรุณาระบุปีหลักสูตร/ฉบับหลักสูตรของ IT ที่ต้องการ";
  assert.equal(displayedText({answer, planClarification: false}), answer);
});

test("program_required preserves actionable program question", () => {
  const answer = "กรุณาระบุหลักสูตรที่ต้องการ";
  assert.equal(displayedText({answer, planClarification: false}), answer);
});

test("context_conflict retains scoped-chat message", () => {
  const answer = "แชทนี้กำหนดไว้สำหรับหลักสูตร IT หากต้องการถาม DSBA โดยตรง ให้ใช้แชทหลักสูตรนั้น";
  assert.equal(displayedText({answer, planClarification: false}), answer);
});

test("missing or unusable backend text retains safe fallback", () => {
  for (const answer of [undefined, null, "", "   ", "ไม่พบคำตอบ"]) {
    assert.equal(displayedText({answer, planClarification: true}), "คำถามนี้ต้องระบุแผนการเรียนก่อน");
    assert.equal(displayedText({answer, planClarification: false}), "ไม่พบคำตอบ");
  }
});

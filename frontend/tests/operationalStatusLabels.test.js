import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

test("provider_unavailable has its own badge label and is not shown as insufficient evidence", () => {
  const page = readFileSync(new URL("../src/pages/ChatPage.jsx", import.meta.url), "utf8");
  assert.match(page, /provider_unavailable:\s*"[^"]*พร้อม[^"]*ลองใหม่[^"]*"/);
  assert.match(page, /insufficient_evidence:\s*"หลักฐานยังไม่เพียงพอ"/);
});

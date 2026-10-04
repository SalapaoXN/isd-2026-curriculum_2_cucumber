import assert from "node:assert/strict";
import test from "node:test";

import {
  buildConversationContext,
  defaultCatalogKey,
  resetContextForEdition,
} from "../src/chatScope.js";

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

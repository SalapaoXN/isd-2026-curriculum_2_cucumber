import unittest
from pathlib import Path
from unittest.mock import patch

from rag.evidence_executor import (
    DirectPrerequisiteBurden,
    DirectPrerequisiteBurdenResult,
    DirectPrerequisiteRequirement,
    EvidenceBundle,
    EvidenceExecutionResult,
    execute_evidence_plan,
)
from rag.evidence_planner import (
    EvidencePlan,
    EvidenceRequest,
    StructuralScope,
    plan_evidence,
)
from rag.qa import _merge_conversation_context
from rag.qa import _compose_evidence_claims, ask
from rag.query_spec import QuerySpec, detect_surface_operations, parse_query_spec
from rag.resolution import QueryContext, ResolutionOutcome
from rag.grounded_answer import compose_grounded_answer


DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


class H20ScopePrerequisiteTests(unittest.TestCase):
    @staticmethod
    def _spec(**changes):
        values = {
            "original_question": "question",
            "normalized_question": "question",
            "program": "IT",
            "plans": ("no_coop",),
            "years": (3,),
            "semesters": (1,),
            "course_codes": (),
            "course_name": None,
            "category": None,
            "topic": None,
            "operations": ("prerequisite",),
            "group_by": (),
            "judgement": "none",
        }
        values.update(changes)
        return QuerySpec(**values)

    @staticmethod
    def _resolution(program="IT", candidates=()):
        references = ()
        if candidates:
            from rag.resolution import CourseReferenceResolution

            references = (
                CourseReferenceResolution(
                    reference_type="course_code",
                    reference="06016420",
                    candidates=tuple(candidates),
                ),
            )
        return ResolutionOutcome("answer", (), program, references)

    @staticmethod
    def _scope(**changes):
        values = {
            "program": "IT",
            "plans": ("no_coop",),
            "years": (3,),
            "semesters": (1,),
            "group_by": (),
        }
        values.update(changes)
        return StructuralScope(**values)

    def test_surface_detector_reuses_prerequisite_vocabulary(self):
        self.assertEqual(
            detect_surface_operations("ตัวไหนมี prerequisite"),
            ("prerequisite",),
        )
        self.assertEqual(
            detect_surface_operations("ตัวไหนมีวิชาบังคับก่อน"),
            ("prerequisite",),
        )

    def test_list_context_and_prerequisite_wording_recover_operation(self):
        spec = parse_query_spec("ตัวไหนมี prerequisite")
        merged = _merge_conversation_context(
            spec,
            QueryContext(
                program="IT",
                years=(3,),
                semesters=(1,),
                operations=("list",),
            ),
        )
        self.assertEqual(merged.operations, ("prerequisite",))
        self.assertEqual(merged.program, "IT")
        self.assertEqual(merged.years, (3,))
        self.assertEqual(merged.semesters, (1,))

    def test_prior_count_context_does_not_activate_h20(self):
        merged = _merge_conversation_context(
            parse_query_spec("ตัวไหนมี prerequisite"),
            QueryContext(program="IT", operations=("count",)),
        )
        self.assertEqual(merged.operations, ())

    def test_bare_then_does_not_activate_h20(self):
        merged = _merge_conversation_context(
            parse_query_spec("แล้ว"),
            QueryContext(program="IT", operations=("list",)),
        )
        self.assertEqual(merged.operations, ())

    def test_exact_course_context_keeps_existing_prerequisite_path(self):
        merged = _merge_conversation_context(
            parse_query_spec("มี prerequisite ไหม"),
            QueryContext(
                program="IT",
                course_code="06016454",
                operations=("identity",),
            ),
        )
        self.assertEqual(merged.operations, ("existence", "prerequisite"))
        self.assertEqual(merged.course_codes, ("06016454",))

    def test_topic_prerequisite_path_remains_topic_dependent(self):
        plan = plan_evidence(
            self._spec(
                topic="database",
                operations=("list", "prerequisite"),
            ),
            self._resolution(),
        )
        self.assertEqual(
            [request.kind for request in plan.requests],
            ["course_set", "topic_matches", "prerequisite_facts"],
        )
        self.assertEqual(plan.requests[-1].depends_on, ("topic_matches",))

    def test_scope_prerequisite_plans_course_set_dependency(self):
        plan = plan_evidence(
            self._spec(),
            self._resolution(),
        )
        self.assertEqual(
            [(request.kind, request.depends_on) for request in plan.requests],
            [
                ("course_set", ()),
                ("prerequisite_facts", ("course_set",)),
            ],
        )

    def _execute_scope_prerequisite(self, course_set_result, *, state_by_id=None):
        scope = self._scope()
        course_set = EvidenceRequest("course_set", "course_set", scope)
        prerequisites = EvidenceRequest(
            "prerequisites",
            "prerequisite_facts",
            scope,
            depends_on=("course_set",),
        )
        plan = EvidencePlan(scope=scope, requests=(course_set, prerequisites))
        state_by_id = state_by_id or {}
        with patch(
            "rag.evidence_executor.scoped_course_set",
            return_value=course_set_result,
        ), patch(
            "rag.evidence_executor.prerequisite_state",
            side_effect=lambda _db_path, course_id: state_by_id[course_id],
        ):
            return execute_evidence_plan(DB_PATH, plan).results[-1]

    def test_executor_expands_normal_and_alternative_members(self):
        provenance = ({"source_page": 1},)
        course_set = {
            "status": "ok",
            "program": "IT",
            "courses": (
                {
                    "program": "IT",
                    "course_id": 101,
                    "course_code": "N101",
                    "is_alternative": False,
                    "provenance": provenance,
                },
                {
                    "program": "IT",
                    "course_id": None,
                    "course_code": None,
                    "is_alternative": True,
                    "alternative_courses": (
                        {
                            "course_id": 201,
                            "course_code": "A201",
                            "provenance": provenance,
                        },
                        {
                            "course_id": 202,
                            "course_code": "A202",
                            "provenance": provenance,
                        },
                    ),
                    "provenance": provenance,
                },
            ),
            "provenance": provenance,
        }
        burden = DirectPrerequisiteBurdenResult(
            status="complete",
            burdens=(
                DirectPrerequisiteBurden(
                    program="IT",
                    course_id=101,
                    course_code="N101",
                    status="complete",
                    required_course_count=0,
                    alternative_group_count=0,
                    alternative_member_counts=(),
                    ordered_requirement_groups=(),
                    provenance=provenance,
                ),
            ),
            provenance=provenance,
        )
        with patch(
            "rag.evidence_executor.build_direct_prerequisite_burden",
            return_value=burden,
        ) as build:
            result = self._execute_scope_prerequisite(course_set)

        build.assert_called_once_with(
            DB_PATH,
            (
                {"program": "IT", "course_id": 101, "course_code": "N101"},
                {"program": "IT", "course_id": 201, "course_code": "A201"},
                {"program": "IT", "course_id": 202, "course_code": "A202"},
            ),
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.payload[0].course_code, "N101")
        self.assertEqual(result.payload[0].provenance, provenance)

    def test_unknown_prerequisite_state_is_insufficient_evidence(self):
        course_set = {
            "status": "ok",
            "courses": ({
                "program": "IT",
                "course_id": 101,
                "course_code": "N101",
                "is_alternative": False,
                "provenance": ({"source_page": 1},),
            },),
            "provenance": ({"source_page": 1},),
        }
        result = self._execute_scope_prerequisite(
            course_set,
            state_by_id={101: {"state": "unknown", "provenance": ({"source_page": 2},)}},
        )
        self.assertEqual(result.status, "insufficient_evidence")

    def test_empty_course_set_is_valid_empty(self):
        result = self._execute_scope_prerequisite(
            {"status": "no_data", "courses": (), "provenance": ({"source_page": 1},)},
        )
        self.assertEqual(result.status, "valid_empty")

    def test_prerequisite_burden_contains_target_identity_and_provenance(self):
        provenance = ({"source_page": 9},)
        course_set = {
            "status": "ok",
            "courses": ({
                "program": "IT",
                "course_id": 101,
                "course_code": "N101",
                "is_alternative": False,
                "provenance": provenance,
            },),
            "provenance": provenance,
        }
        result = self._execute_scope_prerequisite(
            course_set,
            state_by_id={101: {"state": "explicit_none", "provenance": provenance}},
        )
        self.assertEqual(result.status, "complete")
        burden = result.payload[0]
        self.assertEqual(
            (burden.program, burden.course_id, burden.course_code),
            ("IT", 101, "N101"),
        )
        self.assertEqual(burden.provenance, provenance)

    @staticmethod
    def _burden(
        course_id,
        course_code,
        *,
        program="IT",
        required=True,
        page=20,
    ):
        provenance = ({"source_page": page},)
        groups = ()
        if required:
            groups = (
                DirectPrerequisiteRequirement(
                    kind="required_course",
                    requirement_type="required",
                    prerequisite_course_id=900 + course_id,
                    prerequisite_code=f"P{course_code}",
                    prerequisite_name_th=None,
                    prerequisite_name_en=f"Prerequisite {course_code}",
                    alternative_group_id=None,
                    minimum_choices=None,
                    maximum_choices=None,
                    alternative_members=(),
                    provenance=provenance,
                ),
            )
        return DirectPrerequisiteBurden(
            program=program,
            course_id=course_id,
            course_code=course_code,
            status="complete",
            required_course_count=1 if required else 0,
            alternative_group_count=0,
            alternative_member_counts=(),
            ordered_requirement_groups=groups,
            provenance=provenance,
        )

    def _composition_bundle(self, courses, burdens, *, scope=None, prerequisite_status="complete"):
        scope = scope or self._scope()
        course_request = EvidenceRequest("course_set", "course_set", scope)
        prerequisite_request = EvidenceRequest(
            "prerequisite_facts",
            "prerequisite_facts",
            scope,
            depends_on=("course_set",),
        )
        course_provenance = tuple(
            reference
            for course in courses
            for reference in course.get("provenance", ())
        )
        course_result = EvidenceExecutionResult(
            "course_set",
            "course_set",
            course_request,
            scope,
            "complete",
            {"status": "ok", "courses": tuple(courses), "provenance": course_provenance},
        )
        prerequisite_result = EvidenceExecutionResult(
            "prerequisite_facts",
            "prerequisite_facts",
            prerequisite_request,
            scope,
            prerequisite_status,
            tuple(burdens),
        )
        return EvidenceBundle(
            EvidencePlan(scope, (course_request, prerequisite_request)),
            (course_result, prerequisite_result),
        )

    @staticmethod
    def _normal_course(course_id, course_code, page):
        return {
            "program": "IT",
            "course_id": course_id,
            "course_code": course_code,
            "is_alternative": False,
            "alternative_group_id": None,
            "provenance": ({"source_page": page},),
        }

    @staticmethod
    def _alternative_course(group_id, members, page=10):
        return {
            "program": "IT",
            "course_id": None,
            "course_code": None,
            "is_alternative": True,
            "alternative_group_id": group_id,
            "alternative_courses": tuple(members),
            "provenance": ({"source_page": page},),
        }

    def test_scope_prerequisite_composes_only_target_courses(self):
        courses = (
            self._normal_course(101, "A101", 10),
            self._normal_course(102, "A102", 11),
        )
        bundle = self._composition_bundle(
            courses,
            (self._burden(101, "A101"), self._burden(102, "A102", required=False)),
        )
        claims = _compose_evidence_claims(self._spec(), bundle)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].operation, "list")
        self.assertEqual(claims[0].status, "complete")
        self.assertEqual(
            [course["course_code"] for course in claims[0].value],
            ["A101"],
        )
        self.assertNotIn("PA101", repr(claims[0].value))

    def test_all_explicit_none_candidates_are_valid_empty(self):
        courses = (
            self._normal_course(101, "A101", 10),
            self._normal_course(102, "A102", 11),
        )
        bundle = self._composition_bundle(
            courses,
            (
                self._burden(101, "A101", required=False),
                self._burden(102, "A102", required=False),
            ),
        )
        claim = _compose_evidence_claims(self._spec(), bundle)[0]

        self.assertEqual(claim.operation, "list")
        self.assertEqual(claim.status, "valid_empty")
        self.assertEqual(claim.value, ())

    def test_alternative_all_required_keeps_logical_group(self):
        members = (
            {"course_id": 201, "course_code": "A201", "provenance": ({"source_page": 12},)},
            {"course_id": 202, "course_code": "A202", "provenance": ({"source_page": 13},)},
        )
        courses = (self._alternative_course(7, members),)
        bundle = self._composition_bundle(
            courses,
            (self._burden(201, "A201"), self._burden(202, "A202")),
        )
        claim = _compose_evidence_claims(self._spec(), bundle)[0]

        self.assertEqual(claim.status, "complete")
        self.assertEqual(len(claim.value), 1)
        self.assertEqual(claim.value[0]["alternative_group_id"], 7)
        self.assertIsNone(claim.value[0]["course_code"])

    def test_alternative_all_explicit_none_is_excluded(self):
        members = (
            {"course_id": 201, "course_code": "A201", "provenance": ({"source_page": 12},)},
            {"course_id": 202, "course_code": "A202", "provenance": ({"source_page": 13},)},
        )
        bundle = self._composition_bundle(
            (self._alternative_course(7, members),),
            (
                self._burden(201, "A201", required=False),
                self._burden(202, "A202", required=False),
            ),
        )
        claim = _compose_evidence_claims(self._spec(), bundle)[0]

        self.assertEqual(claim.status, "valid_empty")
        self.assertEqual(claim.value, ())

    def test_mixed_alternative_member_states_fail_closed(self):
        members = (
            {"course_id": 201, "course_code": "A201", "provenance": ({"source_page": 12},)},
            {"course_id": 202, "course_code": "A202", "provenance": ({"source_page": 13},)},
        )
        bundle = self._composition_bundle(
            (self._alternative_course(7, members),),
            (self._burden(201, "A201"), self._burden(202, "A202", required=False)),
        )
        claim = _compose_evidence_claims(self._spec(), bundle)[0]

        self.assertEqual(claim.status, "insufficient_evidence")
        self.assertIsNone(claim.value)

    def test_missing_or_duplicate_burden_identity_fails_closed(self):
        course = self._normal_course(101, "A101", 10)
        cases = (
            ("missing", ()),
            ("duplicate", (self._burden(101, "A101"), self._burden(101, "A101"))),
        )
        for label, burdens in cases:
            with self.subTest(label=label):
                bundle = self._composition_bundle((course,), burdens)
                claim = _compose_evidence_claims(self._spec(), bundle)[0]
                self.assertEqual(claim.status, "insufficient_evidence")
                self.assertIsNone(claim.value)

    def test_claim_provenance_covers_course_set_and_excluded_burden(self):
        courses = (
            self._normal_course(101, "A101", 10),
            self._normal_course(102, "A102", 11),
        )
        bundle = self._composition_bundle(
            courses,
            (self._burden(101, "A101", page=20), self._burden(102, "A102", required=False, page=30)),
        )
        claim = _compose_evidence_claims(self._spec(), bundle)[0]
        pages = {reference["source_page"] for reference in claim.provenance}

        self.assertEqual(pages, {10, 11, 20, 30})

    def test_poisoned_partition_prevents_authoritative_partial_answer(self):
        good_scope = self._scope(plans=("coop",))
        bad_scope = self._scope(plans=("no_coop",))
        good_course = EvidenceRequest("courses_coop", "course_set", good_scope)
        good_prereq = EvidenceRequest(
            "prereqs_coop", "prerequisite_facts", good_scope, depends_on=("courses_coop",)
        )
        bad_course = EvidenceRequest("courses_no_coop", "course_set", bad_scope)
        bad_prereq = EvidenceRequest(
            "prereqs_no_coop", "prerequisite_facts", bad_scope, depends_on=("courses_no_coop",)
        )
        good_row = self._normal_course(101, "A101", 10)
        good_course_result = EvidenceExecutionResult(
            "courses_coop", "course_set", good_course, good_scope, "complete",
            {"status": "ok", "courses": (good_row,), "provenance": good_row["provenance"]},
        )
        good_prereq_result = EvidenceExecutionResult(
            "prereqs_coop", "prerequisite_facts", good_prereq, good_scope, "complete",
            (self._burden(101, "A101"),),
        )
        bad_course_result = EvidenceExecutionResult(
            "courses_no_coop", "course_set", bad_course, bad_scope, "complete",
            {"status": "ok", "courses": (self._normal_course(102, "A102", 11),), "provenance": ({"source_page": 11},)},
        )
        bad_prereq_result = EvidenceExecutionResult(
            "prereqs_no_coop", "prerequisite_facts", bad_prereq, bad_scope,
            "insufficient_evidence",
        )
        bundle = EvidenceBundle(
            EvidencePlan(good_scope, (good_course, good_prereq, bad_course, bad_prereq)),
            (good_course_result, good_prereq_result, bad_course_result, bad_prereq_result),
        )
        answer = compose_grounded_answer(
            composed_claims=_compose_evidence_claims(self._spec(), bundle)
        )

        self.assertEqual(answer.status, "insufficient_evidence")

    def test_flagship_it_y3s1_prerequisite_query_remains_insufficient(self):
        first = ask(DB_PATH, "IT ปี 3 เทอม 1 มีวิชาอะไรบ้าง")
        result = ask(DB_PATH, "ตัวไหนมี prerequisite", conversation_context=first["next_context"])

        self.assertEqual(result["result"].status, "insufficient_evidence")

    def test_exact_course_prerequisite_claim_remains_unchanged(self):
        result = ask(DB_PATH, "IT 06016454 มี prerequisite ไหม")

        self.assertEqual(result["result"].status, "answer")
        self.assertIn("prerequisite", [claim.operation for claim in result["result"].claims])


if __name__ == "__main__":
    unittest.main()

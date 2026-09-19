import unittest
from dataclasses import fields
from unittest.mock import patch

from rag.evidence_executor import (
    DirectPrerequisiteBurden,
    DirectPrerequisiteBurdenResult,
    EvidenceExecutionResult,
    build_direct_prerequisite_burden,
    execute_evidence_plan,
)
from rag.evidence_planner import EvidencePlan, EvidenceRequest, StructuralScope
from rag.retrieval.retrieve import ConstrainedTopicRetrievalResult


PROVENANCE = ({"provenance_id": 1, "document_category": "description"},)


def candidate(course_id=10, course_code="C10", program="IT"):
    return {
        "course_id": course_id,
        "course_code": course_code,
        "program": program,
    }


def direct_record(
    prerequisite_course_id=20,
    prerequisite_code="P20",
    *,
    prerequisite_id=1,
    order=1,
    provenance=PROVENANCE,
):
    return {
        "prerequisite_id": prerequisite_id,
        "course_id": 10,
        "prerequisite_course_id": prerequisite_course_id,
        "prerequisite_code": prerequisite_code,
        "prerequisite_name_th": "วิชาก่อน",
        "prerequisite_name_en": "PREREQUISITE",
        "alternative_group_id": None,
        "is_alternative": False,
        "alternative_group": None,
        "alternative_courses": (),
        "prerequisite_order": order,
        "requirement_type": "required",
        "provenance": provenance,
    }


def alternative_record(
    *,
    group_id=7,
    minimum=1,
    maximum=1,
    members=None,
    provenance=PROVENANCE,
    prerequisite_id=2,
):
    members = members or (
        {
            "course_id": 30,
            "course_code": "P30",
            "name_th": "ตัวเลือกหนึ่ง",
            "name_en": "OPTION ONE",
            "provenance": PROVENANCE,
        },
        {
            "course_id": 31,
            "course_code": "P31",
            "name_th": "ตัวเลือกสอง",
            "name_en": "OPTION TWO",
            "provenance": PROVENANCE,
        },
    )
    group = {
        "alternative_group_id": group_id,
        "group_key": "choice",
        "label": "เลือกหนึ่งวิชา",
        "minimum_choices": minimum,
        "maximum_choices": maximum,
        "notes": "หนึ่งในกลุ่ม",
        "alternative_courses": members,
    }
    return {
        "prerequisite_id": prerequisite_id,
        "course_id": 10,
        "prerequisite_course_id": None,
        "prerequisite_code": None,
        "prerequisite_name_th": None,
        "prerequisite_name_en": None,
        "alternative_group_id": group_id,
        "is_alternative": True,
        "alternative_group": group,
        "alternative_courses": members,
        "prerequisite_order": 2,
        "requirement_type": "required",
        "provenance": provenance,
    }


class DirectPrerequisiteBurdenTests(unittest.TestCase):
    def _state(self, state, records=(), provenance=PROVENANCE):
        return {
            "state": state,
            "records": records,
            "prerequisite_text": "ไม่มี" if state == "explicit_none" else None,
            "provenance": provenance,
        }

    def test_explicit_none_is_zero_burden_with_provenance(self):
        with patch(
            "rag.evidence_executor.prerequisite_state",
            return_value=self._state("explicit_none"),
        ):
            result = build_direct_prerequisite_burden("unused", [candidate()])

        burden = result.burdens[0]
        self.assertEqual(result.status, "complete")
        self.assertEqual(burden.required_course_count, 0)
        self.assertEqual(burden.alternative_group_count, 0)
        self.assertEqual(burden.ordered_requirement_groups, ())
        self.assertTrue(burden.provenance)

    def test_one_and_multiple_direct_prerequisites_are_retained_and_counted(self):
        records = (
            direct_record(),
            direct_record(
                prerequisite_id=3,
                prerequisite_course_id=21,
                prerequisite_code="P21",
                order=2,
            ),
        )
        with patch(
            "rag.evidence_executor.prerequisite_state",
            return_value=self._state("required", records),
        ):
            result = build_direct_prerequisite_burden("unused", [candidate()])

        burden = result.burdens[0]
        self.assertEqual(burden.required_course_count, 2)
        self.assertEqual(burden.alternative_group_count, 0)
        self.assertEqual(
            [group.prerequisite_course_id for group in burden.ordered_requirement_groups],
            [20, 21],
        )

    def test_alternative_group_is_one_group_and_preserves_members_and_choices(self):
        record = alternative_record(minimum=2, maximum=2)
        with patch(
            "rag.evidence_executor.prerequisite_state",
            return_value=self._state("required", (record,)),
        ):
            result = build_direct_prerequisite_burden("unused", [candidate()])

        burden = result.burdens[0]
        group = burden.ordered_requirement_groups[0]
        self.assertEqual(burden.required_course_count, 0)
        self.assertEqual(burden.alternative_group_count, 1)
        self.assertEqual(burden.alternative_member_counts, (2,))
        self.assertEqual(group.minimum_choices, 2)
        self.assertEqual(group.maximum_choices, 2)
        self.assertEqual(
            [member["course_code"] for member in group.alternative_members],
            ["P30", "P31"],
        )

    def test_direct_and_alternative_counts_remain_separate(self):
        records = (direct_record(), alternative_record())
        with patch(
            "rag.evidence_executor.prerequisite_state",
            return_value=self._state("required", records),
        ):
            result = build_direct_prerequisite_burden("unused", [candidate()])

        burden = result.burdens[0]
        self.assertEqual(burden.required_course_count, 1)
        self.assertEqual(burden.alternative_group_count, 1)
        self.assertEqual(len(burden.ordered_requirement_groups), 2)

    def test_identical_duplicate_alternative_records_normalize_once(self):
        record = alternative_record()
        duplicate = dict(record, prerequisite_id=99)
        with patch(
            "rag.evidence_executor.prerequisite_state",
            return_value=self._state("required", (record, duplicate)),
        ):
            result = build_direct_prerequisite_burden("unused", [candidate()])

        burden = result.burdens[0]
        self.assertEqual(burden.alternative_group_count, 1)
        self.assertEqual(len(burden.ordered_requirement_groups), 1)

    def test_conflicting_duplicate_alternative_group_fails_closed(self):
        first = alternative_record(minimum=1)
        second = alternative_record(minimum=2, prerequisite_id=99)
        with patch(
            "rag.evidence_executor.prerequisite_state",
            return_value=self._state("required", (first, second)),
        ):
            result = build_direct_prerequisite_burden("unused", [candidate()])

        self.assertEqual(result.status, "insufficient_evidence")

    def test_malformed_alternative_member_fails_closed(self):
        record = alternative_record(
            members=({
                "course_id": 30,
                "course_code": "P30",
                "provenance": (),
            },),
        )
        with patch(
            "rag.evidence_executor.prerequisite_state",
            return_value=self._state("required", (record,)),
        ):
            result = build_direct_prerequisite_burden("unused", [candidate()])

        self.assertEqual(result.status, "insufficient_evidence")

    def test_unknown_raw_only_and_missing_provenance_fail_closed(self):
        for state in (
            self._state("unknown", provenance=PROVENANCE),
            self._state("required", (direct_record(provenance=()),)),
            self._state("explicit_none", provenance=()),
        ):
            with patch("rag.evidence_executor.prerequisite_state", return_value=state):
                result = build_direct_prerequisite_burden("unused", [candidate()])
            self.assertEqual(result.status, "insufficient_evidence")

    def test_all_candidates_must_be_complete(self):
        states = {
            10: self._state("required", (direct_record(),)),
            11: self._state("unknown"),
        }
        with patch(
            "rag.evidence_executor.prerequisite_state",
            side_effect=lambda _db, course_id: states[course_id],
        ):
            result = build_direct_prerequisite_burden(
                "unused",
                [candidate(), candidate(11, "C11")],
            )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.burdens, ())

    def test_multiple_complete_candidates_produce_one_burden_each(self):
        states = {
            10: self._state("required", (direct_record(),)),
            11: self._state("explicit_none"),
        }
        with patch(
            "rag.evidence_executor.prerequisite_state",
            side_effect=lambda _db, course_id: states[course_id],
        ):
            result = build_direct_prerequisite_burden(
                "unused",
                [candidate(), candidate(11, "C11")],
            )

        self.assertEqual(result.status, "complete")
        self.assertEqual([burden.course_id for burden in result.burdens], [10, 11])

    def test_empty_candidate_set_is_valid_empty(self):
        result = build_direct_prerequisite_burden("unused", [])
        self.assertEqual(result.status, "valid_empty")
        self.assertEqual(result.burdens, ())

    def test_duplicate_candidates_are_deterministically_deduplicated(self):
        with patch(
            "rag.evidence_executor.prerequisite_state",
            return_value=self._state("explicit_none"),
        ) as state:
            result = build_direct_prerequisite_burden(
                "unused",
                [candidate(), candidate()],
            )

        self.assertEqual(len(result.burdens), 1)
        state.assert_called_once_with("unused", 10)

    def test_no_transitive_traversal_or_ranking_fields(self):
        with patch(
            "rag.evidence_executor.prerequisite_state",
            return_value=self._state("required", (direct_record(),)),
        ) as state:
            result = build_direct_prerequisite_burden("unused", [candidate()])

        state.assert_called_once_with("unused", 10)
        burden_fields = {field.name for field in fields(result.burdens[0])}
        self.assertFalse(burden_fields & {"few", "many", "score", "rank", "recommendation"})


class TopicPrerequisiteDependencyTests(unittest.TestCase):
    @staticmethod
    def _scope():
        return StructuralScope(
            program="IT",
            plans=("coop",),
            years=(3,),
            semesters=(),
        )

    def _plan(self):
        scope = self._scope()
        return EvidencePlan(
            scope=scope,
            requests=(
                EvidenceRequest("course_set", "course_set", scope),
                EvidenceRequest(
                    "topic_matches",
                    "topic_matches",
                    scope,
                    depends_on=("course_set",),
                    topic="data",
                ),
                EvidenceRequest(
                    "topic_prerequisite_facts",
                    "prerequisite_facts",
                    scope,
                    depends_on=("topic_matches",),
                ),
            ),
        )

    @staticmethod
    def _course_set_result(request, scope):
        candidate = {
            "program": "IT",
            "course_code": "C10",
            "course_id": 10,
            "provenance": PROVENANCE,
        }
        return EvidenceExecutionResult(
            request_id=request.request_id,
            kind=request.kind,
            planned_request=request,
            effective_scope=scope,
            status="complete",
            payload={"courses": (candidate,), "provenance": PROVENANCE},
        )

    def test_complete_topic_candidates_feed_only_grounded_identity_to_builder(self):
        plan = self._plan()
        scope = self._scope()
        candidates = (
            {"program": "IT", "course_code": "C10", "course_id": 10, "score": 0.9},
            {"program": "IT", "course_code": "C11", "course_id": 11, "score": 0.8},
        )
        retrieval = ConstrainedTopicRetrievalResult(
            status="scored",
            candidates=candidates,
            scored_candidates=candidates,
        )
        burden = DirectPrerequisiteBurden(
            program="IT",
            course_id=10,
            course_code="C10",
            status="complete",
            required_course_count=0,
            alternative_group_count=0,
            alternative_member_counts=(),
            ordered_requirement_groups=(),
            provenance=PROVENANCE,
        )

        def execute(_db_path, request, effective_scope):
            return self._course_set_result(request, effective_scope)

        with patch("rag.evidence_executor._execute_request", side_effect=execute), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence",
            return_value=retrieval,
        ), patch(
            "rag.evidence_executor.build_direct_prerequisite_burden",
            return_value=DirectPrerequisiteBurdenResult(
                status="complete",
                burdens=(burden,),
                provenance=PROVENANCE,
            ),
        ) as build:
            results = execute_evidence_plan("unused", plan).results

        build.assert_called_once_with(
            "unused",
            (
                {"program": "IT", "course_code": "C10", "course_id": 10},
                {"program": "IT", "course_code": "C11", "course_id": 11},
            ),
        )
        self.assertEqual(results[-1].status, "complete")
        self.assertEqual(results[-1].effective_scope, scope)
        self.assertEqual(results[-1].payload[0].required_course_count, 0)

    def test_topic_dependency_builds_complete_canonical_burdens(self):
        plan = self._plan()
        candidates = (
            {"program": "IT", "course_code": "C10", "course_id": 10},
            {"program": "IT", "course_code": "C11", "course_id": 11},
        )
        retrieval = ConstrainedTopicRetrievalResult(
            status="scored",
            candidates=candidates,
            scored_candidates=candidates,
        )
        states = {
            10: {
                "state": "explicit_none",
                "records": (),
                "prerequisite_text": "ไม่มี",
                "provenance": PROVENANCE,
            },
            11: {
                "state": "required",
                "records": (direct_record(),),
                "prerequisite_text": None,
                "provenance": PROVENANCE,
            },
        }

        def execute(_db_path, request, effective_scope):
            return self._course_set_result(request, effective_scope)

        with patch("rag.evidence_executor._execute_request", side_effect=execute), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence",
            return_value=retrieval,
        ), patch(
            "rag.evidence_executor.prerequisite_state",
            side_effect=lambda _db, course_id: states[course_id],
        ):
            result = execute_evidence_plan("unused", plan).results[-1]

        self.assertEqual(result.status, "complete")
        self.assertEqual(
            [(burden.course_id, burden.required_course_count) for burden in result.payload],
            [(10, 0), (11, 1)],
        )

    def test_dependency_failure_and_empty_do_not_call_burden_builder(self):
        for retrieval_status, expected_status in (
            ("description_missing", "insufficient_evidence"),
            ("no_threshold_matches", "valid_empty"),
        ):
            with self.subTest(retrieval_status=retrieval_status):
                plan = self._plan()
                scope = self._scope()
                retrieval = ConstrainedTopicRetrievalResult(
                    status=retrieval_status,
                    candidates=(),
                    scored_candidates=(),
                )

                def execute(_db_path, request, effective_scope):
                    return self._course_set_result(request, effective_scope)

                with patch(
                    "rag.evidence_executor._execute_request", side_effect=execute
                ), patch(
                    "rag.evidence_executor.retrieve_constrained_topic_evidence",
                    return_value=retrieval,
                ), patch(
                    "rag.evidence_executor.build_direct_prerequisite_burden"
                ) as build:
                    result = execute_evidence_plan("unused", plan).results[-1]

                self.assertEqual(result.status, expected_status)
                build.assert_not_called()

    def test_malformed_topic_candidate_fails_closed_before_builder(self):
        plan = self._plan()
        retrieval = ConstrainedTopicRetrievalResult(
            status="scored",
            candidates=({"program": "IT", "course_code": "C10"},),
            scored_candidates=({"program": "IT", "course_code": "C10"},),
        )

        def execute(_db_path, request, effective_scope):
            return self._course_set_result(request, effective_scope)

        with patch("rag.evidence_executor._execute_request", side_effect=execute), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence",
            return_value=retrieval,
        ), patch(
            "rag.evidence_executor.build_direct_prerequisite_burden"
        ) as build:
            result = execute_evidence_plan("unused", plan).results[-1]

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.primitive_state, "malformed_topic_dependency")
        build.assert_not_called()

    def test_builder_insufficient_candidate_pack_propagates(self):
        plan = self._plan()
        retrieval = ConstrainedTopicRetrievalResult(
            status="scored",
            candidates=({"program": "IT", "course_code": "C10", "course_id": 10},),
            scored_candidates=({"program": "IT", "course_code": "C10", "course_id": 10},),
        )

        def execute(_db_path, request, effective_scope):
            return self._course_set_result(request, effective_scope)

        with patch("rag.evidence_executor._execute_request", side_effect=execute), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence",
            return_value=retrieval,
        ), patch(
            "rag.evidence_executor.build_direct_prerequisite_burden",
            return_value=DirectPrerequisiteBurdenResult(
                status="insufficient_evidence",
                primitive_state="prerequisite_burden_incomplete",
            ),
        ):
            result = execute_evidence_plan("unused", plan).results[-1]

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.primitive_state, "prerequisite_burden_incomplete")


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
from unittest.mock import call, patch

from rag.evidence_executor import (
    EvidenceBundle,
    EvidenceExecutionResult,
    execute_exact_similarity_from_bundle,
    execute_evidence_plan,
)
from rag.evidence_planner import EvidencePlan, EvidenceRequest, StructuralScope
from rag.retrieval.retrieve import ConstrainedTopicRetrievalResult, SimilarityEvidence


DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "cucumber_outputs"
    / "runtime"
    / "curriculum.db"
)


class EvidenceExecutorTests(unittest.TestCase):
    @staticmethod
    def _scope(**changes):
        values = {
            "program": "IT",
            "plans": ("coop",),
            "years": (2,),
            "semesters": (1,),
            "group_by": (),
        }
        values.update(changes)
        return StructuralScope(**values)

    @staticmethod
    def _plan(*requests, scope=None):
        scope = scope or requests[0].scope
        return EvidencePlan(scope=scope, requests=tuple(requests))

    @staticmethod
    def _course_set_payload(plan_key="coop"):
        return {
            "status": "ok",
            "program": "IT",
            "plan_keys": (plan_key,),
            "courses": ({
                "course_id": 20,
                "program": "IT",
                "course_code": "06016420",
                "plan_key": plan_key,
                "provenance": ({"source_page": 1},),
            },),
        }

    def _topic_plan(self, scope=None):
        scope = scope or self._scope()
        course_set = EvidenceRequest("courses", "course_set", scope)
        topic = EvidenceRequest(
            "topics",
            "topic_matches",
            scope,
            depends_on=("courses",),
            topic="database",
        )
        return self._plan(course_set, topic, scope=scope)

    @staticmethod
    def _complete_result(request, scope, payload):
        return EvidenceExecutionResult(
            request_id=request.request_id,
            kind=request.kind,
            planned_request=request,
            effective_scope=scope,
            status="complete",
            payload=payload,
        )

    def test_concrete_course_set_is_typed_and_provenanced(self):
        scope = self._scope()
        request = EvidenceRequest("courses", "course_set", scope)

        bundle = execute_evidence_plan(DB_PATH, self._plan(request))

        self.assertIsInstance(bundle, EvidenceBundle)
        result = bundle.results[0]
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.effective_scope.plans, ("coop",))
        self.assertTrue(result.payload["courses"])
        self.assertTrue(result.payload["courses"][0]["provenance"])

    def test_applicable_plans_are_separate_results(self):
        scope = self._scope(
            plans=(),
            years=(),
            semesters=(),
            expand_applicable=("plan",),
            unconstrained=("year", "semester", "course"),
        )
        request = EvidenceRequest("courses", "course_set", scope)

        results = execute_evidence_plan(DB_PATH, self._plan(request)).results

        self.assertEqual(
            [result.effective_scope.plans for result in results],
            [("coop",), ("no_coop",)],
        )
        self.assertEqual(len({result.effective_scope.plans for result in results}), 2)

    def test_request_course_targets_narrow_materialized_scopes(self):
        left = {
            "course_id": 632,
            "program": "IT",
            "course_code": "06016414",
        }
        right = {
            "course_id": 634,
            "program": "IT",
            "course_code": "06016419",
        }
        scope = self._scope(
            plans=(),
            years=(),
            semesters=(),
            expand_applicable=("plan",),
            unconstrained=("year", "semester"),
            course_targets=(left, right),
        )
        request = EvidenceRequest(
            "left_description",
            "description_evidence",
            scope,
            course_targets=(left,),
        )

        results = execute_evidence_plan(DB_PATH, self._plan(request)).results

        self.assertEqual([result.effective_scope.plans for result in results], [
            ("coop",),
            ("no_coop",),
        ])
        for result in results:
            self.assertEqual(result.effective_scope.course_targets, (left,))
            self.assertTrue(result.payload)
            expected_course_id = 632 if result.effective_scope.plans == ("coop",) else 736
            self.assertEqual(
                {evidence["course_id"] for evidence in result.payload},
                {expected_course_id},
            )
            self.assertTrue(all(
                evidence["course_code"] == left["course_code"]
                for evidence in result.payload
            ))

    def test_each_applicable_similarity_description_side_stays_single_course(self):
        left = {
            "course_id": 632,
            "program": "IT",
            "course_code": "06016414",
        }
        right = {
            "course_id": 634,
            "program": "IT",
            "course_code": "06016419",
        }
        scope = self._scope(
            plans=(),
            years=(),
            semesters=(),
            expand_applicable=("plan",),
            unconstrained=("year", "semester"),
            course_targets=(left, right),
        )
        requests = (
            EvidenceRequest(
                "left_description",
                "description_evidence",
                scope,
                course_targets=(left,),
            ),
            EvidenceRequest(
                "right_description",
                "description_evidence",
                scope,
                course_targets=(right,),
            ),
        )

        results = execute_evidence_plan(DB_PATH, self._plan(*requests)).results

        for request_id, expected_code in (
            ("left_description", left["course_code"]),
            ("right_description", right["course_code"]),
        ):
            request_results = [r for r in results if r.request_id == request_id]
            self.assertEqual(len(request_results), 2)
            for result in request_results:
                self.assertEqual(
                    result.effective_scope.course_targets,
                    (left if expected_code == left["course_code"] else right,),
                )
                expected_course_id = {
                    ("06016414", "coop"): 632,
                    ("06016414", "no_coop"): 736,
                    ("06016419", "coop"): 634,
                    ("06016419", "no_coop"): 738,
                }[(expected_code, result.effective_scope.plans[0])]
                self.assertEqual(
                    {evidence["course_id"] for evidence in result.payload},
                    {expected_course_id},
                )
                self.assertTrue(all(
                    evidence["course_code"] == expected_code
                    for evidence in result.payload
                ))

    def test_unconstrained_year_and_semester_materialize_without_splitting(self):
        scope = self._scope(
            years=(),
            semesters=(),
            unconstrained=("year", "semester", "course"),
        )
        request = EvidenceRequest("courses", "course_set", scope)

        results = execute_evidence_plan(DB_PATH, self._plan(request)).results

        self.assertEqual(len(results), 1)
        effective = results[0].effective_scope
        self.assertEqual(effective.years, (1, 2, 3, 4))
        self.assertEqual(effective.semesters, (1, 2))
        self.assertEqual(effective.unconstrained, ("course",))

    def test_unconstrained_course_remains_one_relation_axis(self):
        scope = self._scope(
            years=(),
            semesters=(),
            unconstrained=("year", "semester", "course"),
        )
        request = EvidenceRequest("courses", "course_set", scope)

        result = execute_evidence_plan(DB_PATH, self._plan(request)).results[0]

        self.assertEqual(result.effective_scope.unconstrained, ("course",))
        self.assertGreater(len(result.payload["courses"]), 1)

    def test_grouped_year_materializes_one_result_per_year(self):
        scope = self._scope(
            years=(),
            semesters=(),
            group_by=("year",),
            unconstrained=("semester", "course"),
        )
        request = EvidenceRequest("courses", "course_set", scope)

        results = execute_evidence_plan(DB_PATH, self._plan(request)).results

        self.assertEqual([r.effective_scope.years for r in results], [(1,), (2,), (3,), (4,)])
        self.assertTrue(all("semester" not in r.effective_scope.unconstrained for r in results))
        self.assertTrue(all(r.effective_scope.unconstrained == ("course",) for r in results))

    def test_grouped_semester_materializes_one_result_per_semester(self):
        scope = self._scope(
            years=(),
            semesters=(),
            group_by=("semester",),
            unconstrained=("year", "course"),
        )
        request = EvidenceRequest("courses", "course_set", scope)

        results = execute_evidence_plan(DB_PATH, self._plan(request)).results

        self.assertEqual([r.effective_scope.semesters for r in results], [(1,), (2,)])
        self.assertTrue(all(r.effective_scope.years == (1, 2, 3, 4) for r in results))
        self.assertTrue(all(r.effective_scope.unconstrained == ("course",) for r in results))

    def test_ungrouped_explicit_axes_stay_together(self):
        scope = self._scope(
            years=(2, 3),
            semesters=(1, 2),
            group_by=(),
        )
        request = EvidenceRequest("courses", "course_set", scope)

        results = execute_evidence_plan(DB_PATH, self._plan(request)).results

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].effective_scope.years, (2, 3))
        self.assertEqual(results[0].effective_scope.semesters, (1, 2))

    def test_grouped_year_and_semester_keep_partitions(self):
        scope = self._scope(
            years=(2, 3),
            semesters=(1, 2),
            group_by=("year", "semester"),
        )
        request = EvidenceRequest("courses", "course_set", scope)

        results = execute_evidence_plan(DB_PATH, self._plan(request)).results
        scopes = [
            (result.effective_scope.plans,
             result.effective_scope.years,
             result.effective_scope.semesters)
            for result in results
        ]

        self.assertEqual(len(scopes), len(set(scopes)))
        self.assertEqual(set(scopes), {
            (("coop",), (2,), (1,)),
            (("coop",), (2,), (2,)),
            (("coop",), (3,), (1,)),
            (("coop",), (3,), (2,)),
        })

    def test_explicit_plan_values_never_merge(self):
        scope = self._scope(
            plans=("coop", "no_coop"),
            years=(2, 3),
            group_by=("plan", "year"),
        )
        request = EvidenceRequest("courses", "course_set", scope)

        results = execute_evidence_plan(DB_PATH, self._plan(request)).results

        self.assertEqual(
            {(r.effective_scope.plans, r.effective_scope.years) for r in results},
            {
                (("coop",), (2,)), (("coop",), (3,)),
                (("no_coop",), (2,)), (("no_coop",), (3,)),
            },
        )

    def test_valid_empty_is_not_insufficient(self):
        scope = self._scope()
        request = EvidenceRequest("courses", "course_set", scope)
        empty = {"status": "no_data", "courses": [], "provenance": []}

        with patch("rag.evidence_executor.scoped_course_set", return_value=empty):
            result = execute_evidence_plan(DB_PATH, self._plan(request)).results[0]

        self.assertEqual(result.status, "valid_empty")
        self.assertEqual(result.primitive_state, "empty_relation")

    def test_credit_facts_keep_authoritative_components_and_provenance(self):
        scope = self._scope()
        request = EvidenceRequest("credits", "credit_facts", scope)

        result = execute_evidence_plan(DB_PATH, self._plan(request)).results[0]

        self.assertIn(result.status, ("complete", "valid_empty"))
        if result.status == "complete":
            self.assertTrue(
                any(
                    "counted_credit_units" in component
                    for component in result.payload["components"]
                )
            )
            self.assertTrue(
                any(component["provenance"] for component in result.payload["components"])
            )

    def test_topic_credit_uses_only_matched_course_targets(self):
        scope = self._scope()
        course_a = {
            "course_id": 20,
            "program": "IT",
            "course_code": "06016420",
            "plan_key": "coop",
            "provenance": ({"source_page": 1},),
        }
        course_b = {
            "course_id": 21,
            "program": "IT",
            "course_code": "06016421",
            "plan_key": "coop",
            "provenance": ({"source_page": 2},),
        }
        courses = EvidenceRequest("courses", "course_set", scope)
        topics = EvidenceRequest(
            "topics",
            "topic_matches",
            scope,
            depends_on=("courses",),
            topic="database",
        )
        credits = EvidenceRequest(
            "credits",
            "credit_facts",
            scope,
            depends_on=("topics",),
        )
        plan = self._plan(courses, topics, credits, scope=scope)

        credit_payload = {
            "status": "ok",
            "components": ({
                "course_id": course_a["course_id"],
                "counted_credit_units": 3,
                "provenance": course_a["provenance"],
            },),
            "provenance": course_a["provenance"],
        }

        def credit_result(*_args, **kwargs):
            self.assertEqual(kwargs["course_targets"], (course_a,))
            return credit_payload

        retrieval = ConstrainedTopicRetrievalResult(
            status="scored",
            candidates=(course_a, course_b),
            scored_candidates=(course_a,),
        )
        with patch(
            "rag.evidence_executor.scoped_course_set",
            return_value={
                "status": "ok",
                "courses": (course_a, course_b),
                "provenance": course_a["provenance"],
            },
        ), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence",
            return_value=retrieval,
        ), patch(
            "rag.evidence_executor.get_semester_credits",
            side_effect=credit_result,
        ) as get_credits:
            result = execute_evidence_plan(DB_PATH, plan).results[-1]

        self.assertEqual(result.status, "complete")
        self.assertEqual(
            [component["course_id"] for component in result.payload["components"]],
            [course_a["course_id"]],
        )
        self.assertEqual(result.payload["components"][0]["provenance"], course_a["provenance"])
        get_credits.assert_called_once()

    def test_non_topic_credit_keeps_full_structural_scope_call(self):
        scope = self._scope()
        request = EvidenceRequest("credits", "credit_facts", scope)
        payload = {
            "status": "ok",
            "components": ({"counted_credit_units": 3, "provenance": ({"source_page": 1},)},),
            "provenance": ({"source_page": 1},),
        }

        with patch(
            "rag.evidence_executor.get_semester_credits",
            return_value=payload,
        ) as get_credits:
            result = execute_evidence_plan(DB_PATH, self._plan(request)).results[0]

        self.assertEqual(result.status, "complete")
        get_credits.assert_called_once_with(DB_PATH, "IT", "coop", 2, 1)

    def test_topic_credit_fanout_keeps_plan_targets_isolated(self):
        scope = self._scope(
            plans=(),
            expand_applicable=("plan",),
        )
        courses = EvidenceRequest("courses", "course_set", scope)
        topics = EvidenceRequest(
            "topics",
            "topic_matches",
            scope,
            depends_on=("courses",),
            topic="database",
        )
        credits = EvidenceRequest(
            "credits",
            "credit_facts",
            scope,
            depends_on=("topics",),
        )
        plan = self._plan(courses, topics, credits, scope=scope)
        calls = []

        def course_set(_db_path, _program, plan_keys, **_kwargs):
            plan_key = tuple(plan_keys)[0]
            course_id = 20 if plan_key == "coop" else 21
            provenance = ({"source_page": course_id},)
            candidate = {
                "course_id": course_id,
                "program": "IT",
                "course_code": f"06016{course_id:03d}",
                "plan_key": plan_key,
                "provenance": provenance,
            }
            return {"status": "ok", "courses": (candidate,), "provenance": provenance}

        def credit_result(_db_path, _program, plan_key, _year, _semester, *, course_targets):
            calls.append((plan_key, tuple(target["plan_key"] for target in course_targets)))
            provenance = ({"source_page": 20 if plan_key == "coop" else 21},)
            return {
                "status": "ok",
                "components": ({"counted_credit_units": 3, "provenance": provenance},),
                "provenance": provenance,
            }

        def topic_result(_db_path, _topic, candidates):
            return ConstrainedTopicRetrievalResult(
                status="scored",
                candidates=tuple(candidates),
                scored_candidates=tuple(candidates),
            )

        with patch(
            "rag.evidence_executor.scoped_course_set",
            side_effect=course_set,
        ), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence",
            side_effect=topic_result,
        ), patch(
            "rag.evidence_executor.get_semester_credits",
            side_effect=credit_result,
        ):
            bundle = execute_evidence_plan(DB_PATH, plan)

        credit_results = [result for result in bundle.results if result.kind == "credit_facts"]
        self.assertEqual([result.effective_scope.plans for result in credit_results], [("coop",), ("no_coop",)])
        self.assertEqual(calls, [("coop", ("coop",)), ("no_coop", ("no_coop",))])

    def test_prerequisite_facts_execute_without_flattening(self):
        scope = self._scope(
            course_targets=({"course_id": 20, "program": "IT", "course_code": "06016420"},),
        )
        request = EvidenceRequest(
            "prereqs", "prerequisite_facts", scope, course_targets=scope.course_targets
        )

        result = execute_evidence_plan(DB_PATH, self._plan(request)).results[0]

        self.assertIn(result.status, ("complete", "valid_empty"))
        if result.status == "complete":
            self.assertIsInstance(result.payload, tuple)

    def test_direct_description_evidence_preserves_partition(self):
        scope = self._scope(
            course_targets=({"course_id": 20, "program": "IT", "course_code": "06016420"},),
        )
        request = EvidenceRequest(
            "description", "description_evidence", scope, course_targets=scope.course_targets
        )
        evidence = [{
            "chunk_id": "desc-20",
            "chunk_type": "description",
            "text": "description",
            "provenance": [{"source_page": 1}],
        }]

        with patch(
            "rag.evidence_executor.fetch_course_description_evidence",
            return_value=evidence,
        ), patch(
            "rag.evidence_executor.scoped_course_set",
            return_value={
                "status": "ok",
                "courses": ({
                    "course_id": 20,
                    "program": "IT",
                    "course_code": "06016420",
                },),
            },
        ):
            result = execute_evidence_plan(DB_PATH, self._plan(request)).results[0]

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.payload[0]["partition"]["plans"], ("coop",))
        self.assertEqual(result.payload[0]["provenance"], ({"source_page": 1},))

    def test_missing_direct_description_is_insufficient(self):
        scope = self._scope(
            course_targets=({"course_id": 20, "program": "IT", "course_code": "06016420"},),
        )
        request = EvidenceRequest(
            "description", "description_evidence", scope, course_targets=scope.course_targets
        )

        with patch(
            "rag.evidence_executor.fetch_course_description_evidence", return_value=[]
        ), patch(
            "rag.evidence_executor.scoped_course_set",
            return_value={
                "status": "ok",
                "courses": ({
                    "course_id": 20,
                    "program": "IT",
                    "course_code": "06016420",
                },),
            },
        ):
            result = execute_evidence_plan(DB_PATH, self._plan(request)).results[0]

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.primitive_state, "description_missing")

    def test_invalid_dependency_cycle_fails_closed(self):
        scope = self._scope()
        first = EvidenceRequest("a", "course_set", scope, depends_on=("b",))
        second = EvidenceRequest("b", "placement_facts", scope, depends_on=("a",))

        bundle = execute_evidence_plan(DB_PATH, self._plan(first, second))

        self.assertEqual([r.status for r in bundle.results], [
            "insufficient_evidence", "insufficient_evidence"
        ])
        self.assertTrue(all(r.primitive_state == "invalid_dependency_graph" for r in bundle.results))

    def test_topic_matches_without_course_set_dependency_fails_closed(self):
        scope = self._scope()
        request = EvidenceRequest("topic", "topic_matches", scope, topic="database")

        with patch("rag.evidence_executor.retrieve_constrained_topic_evidence") as retrieve:
            result = execute_evidence_plan(DB_PATH, self._plan(request)).results[0]

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.primitive_state, "invalid_course_set_dependency")
        retrieve.assert_not_called()

    def test_topic_matches_consumes_only_declared_course_set(self):
        plan = self._topic_plan()
        candidate = self._course_set_payload()["courses"][0]
        retrieval = ConstrainedTopicRetrievalResult(
            status="scored",
            candidates=(candidate,),
            scored_candidates=(candidate,),
        )

        with patch(
            "rag.evidence_executor.scoped_course_set",
            return_value=self._course_set_payload(),
        ), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence",
            return_value=retrieval,
        ) as retrieve:
            results = execute_evidence_plan(DB_PATH, plan).results

        topic_result = results[1]
        self.assertEqual(topic_result.status, "complete")
        self.assertEqual(topic_result.primitive_state, "scored")
        self.assertIs(topic_result.payload, retrieval)
        retrieve.assert_called_once_with(DB_PATH, "database", (candidate,))

    def test_applicable_plan_topic_results_remain_separate(self):
        scope = self._scope(plans=("coop", "no_coop"))
        plan = self._topic_plan(scope)

        def course_set(_db_path, _program, plan_keys, **_filters):
            return self._course_set_payload(plan_keys[0])

        retrievals = [
            ConstrainedTopicRetrievalResult(status="scored", scored_candidates=()),
            ConstrainedTopicRetrievalResult(status="scored", scored_candidates=()),
        ]
        with patch("rag.evidence_executor.scoped_course_set", side_effect=course_set), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence",
            side_effect=retrievals,
        ) as retrieve:
            results = execute_evidence_plan(DB_PATH, plan).results

        topic_results = results[2:]
        self.assertEqual(
            [result.effective_scope.plans for result in topic_results],
            [("coop",), ("no_coop",)],
        )
        self.assertEqual(
            [call.args[2][0]["plan_key"] for call in retrieve.call_args_list],
            ["coop", "no_coop"],
        )

    def test_empty_course_set_maps_to_empty_structural_candidates(self):
        plan = self._topic_plan()
        empty = {"status": "no_data", "courses": (), "provenance": ()}

        with patch("rag.evidence_executor.scoped_course_set", return_value=empty), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence"
        ) as retrieve:
            topic_result = execute_evidence_plan(DB_PATH, plan).results[1]

        self.assertEqual(topic_result.status, "valid_empty")
        self.assertEqual(topic_result.primitive_state, "empty_structural_candidates")
        retrieve.assert_not_called()

    def test_topic_retrieval_states_map_without_losing_wrapper(self):
        states = (
            ("no_threshold_matches", "valid_empty"),
            ("description_missing", "insufficient_evidence"),
            ("vector_missing_or_invalid", "insufficient_evidence"),
        )
        for retrieval_state, expected_status in states:
            with self.subTest(retrieval_state=retrieval_state):
                plan = self._topic_plan()
                retrieval = ConstrainedTopicRetrievalResult(status=retrieval_state)
                with patch(
                    "rag.evidence_executor.scoped_course_set",
                    return_value=self._course_set_payload(),
                ), patch(
                    "rag.evidence_executor.retrieve_constrained_topic_evidence",
                    return_value=retrieval,
                ):
                    topic_result = execute_evidence_plan(DB_PATH, plan).results[1]
                self.assertEqual(topic_result.status, expected_status)
                self.assertEqual(topic_result.primitive_state, retrieval_state)
                self.assertIs(topic_result.payload, retrieval)

    def test_lexical_rescue_with_no_distance_remains_complete(self):
        plan = self._topic_plan()
        candidate = self._course_set_payload()["courses"][0]
        rescued = dict(candidate)
        rescued["description_evidence"] = ({
            "chunk_id": "description-20",
            "distance": None,
            "provenance": ({"source_page": 2},),
        },)
        retrieval = ConstrainedTopicRetrievalResult(
            status="scored",
            candidates=(rescued,),
            scored_candidates=(rescued,),
        )

        with patch("rag.evidence_executor.scoped_course_set", return_value=self._course_set_payload()), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence",
            return_value=retrieval,
        ):
            topic_result = execute_evidence_plan(DB_PATH, plan).results[1]

        self.assertEqual(topic_result.status, "complete")
        self.assertIsNone(topic_result.payload.scored_candidates[0]["description_evidence"][0]["distance"])

    def test_malformed_or_wrong_topic_dependency_fails_closed(self):
        scope = self._scope()
        placement = EvidenceRequest("placements", "placement_facts", scope)
        topic = EvidenceRequest(
            "topics", "topic_matches", scope, depends_on=("placements",), topic="database"
        )
        plan = self._plan(placement, topic)

        with patch("rag.evidence_executor.scoped_course_set", return_value=self._course_set_payload()), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence"
        ) as retrieve:
            topic_result = execute_evidence_plan(DB_PATH, plan).results[1]

        self.assertEqual(topic_result.status, "insufficient_evidence")
        self.assertEqual(topic_result.primitive_state, "invalid_course_set_dependency")
        retrieve.assert_not_called()

    def test_malformed_course_set_candidates_fail_closed(self):
        plan = self._topic_plan()
        malformed = {
            "status": "ok",
            "courses": (None,),
            "provenance": ({"source_page": 1},),
        }

        with patch("rag.evidence_executor.scoped_course_set", return_value=malformed), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence"
        ) as retrieve:
            topic_result = execute_evidence_plan(DB_PATH, plan).results[1]

        self.assertEqual(topic_result.status, "insufficient_evidence")
        self.assertEqual(topic_result.primitive_state, "malformed_course_set_dependency")
        retrieve.assert_not_called()

    def test_direct_partition_failure_isolated_and_independent_request_survives(self):
        scope = self._scope(plans=("coop", "no_coop"))
        courses = EvidenceRequest("courses", "course_set", scope)
        placement = EvidenceRequest("placements", "placement_facts", scope)
        plan = self._plan(courses, placement, scope=scope)

        def execute(_db_path, request, effective_scope):
            if (
                request.request_id == "courses"
                and effective_scope.plans == ("no_coop",)
            ):
                raise OSError("synthetic partition failure")
            provenance = ({"source": request.request_id},)
            return self._complete_result(
                request,
                effective_scope,
                {"value": request.request_id, "provenance": provenance},
            )

        with patch("rag.evidence_executor._execute_request", side_effect=execute):
            results = execute_evidence_plan(DB_PATH, plan).results

        self.assertEqual(
            [(result.request_id, result.effective_scope.plans, result.status) for result in results],
            [
                ("courses", ("coop",), "complete"),
                ("courses", ("no_coop",), "insufficient_evidence"),
                ("placements", ("coop",), "complete"),
                ("placements", ("no_coop",), "complete"),
            ],
        )
        self.assertEqual(results[1].primitive_state, "execution_failure")
        self.assertIsNone(results[1].payload)
        self.assertEqual(results[0].payload["provenance"], ({"source": "courses"},))
        self.assertEqual(results[2].payload["value"], "placements")

    def test_topic_fanout_keeps_valid_sibling_when_dependency_partition_fails(self):
        scope = self._scope(plans=("coop", "no_coop"))
        courses = EvidenceRequest("courses", "course_set", scope)
        topic = EvidenceRequest(
            "topics",
            "topic_matches",
            scope,
            depends_on=("courses",),
            topic="database",
        )
        plan = self._plan(courses, topic, scope=scope)

        def execute(_db_path, request, effective_scope):
            if effective_scope.plans == ("no_coop",):
                raise OSError("synthetic partition failure")
            candidate = {
                "course_id": 20,
                "program": "IT",
                "course_code": "06016420",
                "plan_key": "coop",
                "provenance": ({"source_page": 1},),
            }
            return self._complete_result(
                request,
                effective_scope,
                {"status": "ok", "courses": (candidate,), "provenance": candidate["provenance"]},
            )

        candidate = {
            "course_id": 20,
            "program": "IT",
            "course_code": "06016420",
            "plan_key": "coop",
            "provenance": ({"source_page": 1},),
        }
        retrieval = ConstrainedTopicRetrievalResult(
            status="scored", candidates=(candidate,), scored_candidates=(candidate,)
        )
        with patch("rag.evidence_executor._execute_request", side_effect=execute), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence",
            return_value=retrieval,
        ) as retrieve:
            results = execute_evidence_plan(DB_PATH, plan).results

        self.assertEqual(
            [(result.request_id, result.effective_scope.plans, result.status) for result in results],
            [
                ("courses", ("coop",), "complete"),
                ("courses", ("no_coop",), "insufficient_evidence"),
                ("topics", ("coop",), "complete"),
                ("topics", ("no_coop",), "insufficient_evidence"),
            ],
        )
        self.assertEqual(results[3].primitive_state, "execution_failure")
        retrieve.assert_called_once_with(DB_PATH, "database", (candidate,))

    def test_mixed_six_primitive_plan_coexists_without_flattening(self):
        scope = self._scope()
        requests = (
            EvidenceRequest("courses", "course_set", scope),
            EvidenceRequest("placements", "placement_facts", scope),
            EvidenceRequest("credits", "credit_facts", scope),
            EvidenceRequest("prereqs", "prerequisite_facts", scope),
            EvidenceRequest("descriptions", "description_evidence", scope),
        )
        topic = EvidenceRequest(
            "topics",
            "topic_matches",
            scope,
            depends_on=("courses",),
            topic="database",
        )
        plan = self._plan(*requests, topic, scope=scope)
        candidate = {
            "course_id": 20,
            "program": "IT",
            "course_code": "06016420",
            "plan_key": "coop",
            "provenance": ({"source_page": 1},),
        }

        def execute(_db_path, request, effective_scope):
            payloads = {
                "course_set": {"courses": (candidate,), "provenance": candidate["provenance"]},
                "placement_facts": {"placements": (), "provenance": candidate["provenance"]},
                "credit_facts": {"components": (), "provenance": candidate["provenance"]},
                "prerequisite_facts": ({"course_id": 20, "provenance": candidate["provenance"]},),
                "description_evidence": ({"chunk_id": "desc-20", "provenance": candidate["provenance"]},),
            }
            return self._complete_result(request, effective_scope, payloads[request.kind])

        retrieval = ConstrainedTopicRetrievalResult(
            status="scored", candidates=(candidate,), scored_candidates=(candidate,)
        )
        with patch("rag.evidence_executor._execute_request", side_effect=execute), patch(
            "rag.evidence_executor.retrieve_constrained_topic_evidence",
            return_value=retrieval,
        ):
            bundle = execute_evidence_plan(DB_PATH, plan)

        self.assertIs(bundle.plan, plan)
        self.assertEqual(
            [result.kind for result in bundle.results],
            [
                "course_set",
                "placement_facts",
                "credit_facts",
                "prerequisite_facts",
                "description_evidence",
                "topic_matches",
            ],
        )
        self.assertEqual(bundle.results[-1].status, "complete")
        self.assertIs(bundle.results[-1].payload, retrieval)
        self.assertEqual(bundle.results[0].payload["courses"][0]["course_code"], "06016420")
        self.assertEqual(bundle.results[2].payload["components"], ())
        self.assertEqual(bundle.results[4].payload[0]["chunk_id"], "desc-20")

    def test_plan_and_requests_are_retained_unchanged(self):
        scope = self._scope()
        request = EvidenceRequest("courses", "course_set", scope)
        plan = self._plan(request)

        result = execute_evidence_plan(DB_PATH, plan).results[0]

        self.assertIs(result.planned_request, request)
        self.assertIs(result.planned_request.scope, scope)
        self.assertEqual(plan.requests, (request,))
        with self.assertRaises(TypeError):
            result.payload["courses"] = ()

    def test_malformed_dependency_is_rejected_by_plan_contract(self):
        scope = self._scope()
        request = EvidenceRequest("a", "course_set", scope, depends_on=("missing",))

        with self.assertRaises(ValueError):
            self._plan(request)

    @staticmethod
    def _similarity_bundle(scopes):
        left_target = {
            "course_id": 1,
            "program": "IT",
            "course_code": "00000001",
        }
        right_target = {
            "course_id": 2,
            "program": "IT",
            "course_code": "00000002",
        }
        left_request = EvidenceRequest(
            "left_descriptions",
            "description_evidence",
            scopes[0],
            course_targets=(left_target,),
        )
        right_request = EvidenceRequest(
            "right_descriptions",
            "description_evidence",
            scopes[0],
            course_targets=(right_target,),
        )
        results = []
        for scope in scopes:
            for request, target, text in (
                (left_request, left_target, "left description"),
                (right_request, right_target, "right description"),
            ):
                evidence = {
                    "chunk_id": f"{target['course_code']}-{scope.plans[0]}",
                    "chunk_type": "description",
                    "course_id": target["course_id"],
                    "course_code": target["course_code"],
                    "program": target["program"],
                    "text": text,
                    "partition": {
                        "program": scope.program,
                        "plans": scope.plans,
                        "years": scope.years,
                        "semesters": scope.semesters,
                        "category": scope.category,
                        "group_by": scope.group_by,
                    },
                    "provenance": ({"source_page": 1},),
                }
                results.append(
                    EvidenceExecutionResult(
                        request_id=request.request_id,
                        kind=request.kind,
                        planned_request=request,
                        effective_scope=scope,
                        status="complete",
                        payload=(evidence,),
                    )
                )
        plan = EvidencePlan(scope=scopes[0], requests=(left_request, right_request))
        return EvidenceBundle(plan=plan, results=tuple(results))

    def test_exact_similarity_bridge_calls_aggregate_once_with_existing_payloads(self):
        scope = self._scope()
        bundle = self._similarity_bundle((scope,))
        expected = SimilarityEvidence(status="valid_empty")

        with patch(
            "rag.evidence_executor.aggregate_exact_course_similarity",
            return_value=expected,
        ) as aggregate, patch(
            "rag.evidence_executor.fetch_course_description_evidence"
        ) as fetch:
            result = execute_exact_similarity_from_bundle(
                DB_PATH,
                bundle,
                "left_descriptions",
                "right_descriptions",
            )

        self.assertIs(result, expected)
        aggregate.assert_called_once()
        fetch.assert_not_called()
        left_records, right_records = aggregate.call_args.args[1:3]
        self.assertEqual(left_records[0]["description_evidence"][0]["chunk_id"], "00000001-coop")
        self.assertEqual(right_records[0]["course_code"], "00000002")
        self.assertEqual(left_records[0]["partition"]["plan"], "coop")
        self.assertEqual(left_records[0]["partition"]["plans"], ("coop",))

    def test_exact_similarity_bridge_preserves_each_materialized_partition(self):
        scopes = (self._scope(), self._scope(plans=("no_coop",)))
        bundle = self._similarity_bundle(scopes)
        expected = SimilarityEvidence(status="valid_empty")

        with patch(
            "rag.evidence_executor.aggregate_exact_course_similarity",
            return_value=expected,
        ) as aggregate:
            result = execute_exact_similarity_from_bundle(
                DB_PATH,
                bundle,
                "left_descriptions",
                "right_descriptions",
            )

        self.assertIs(result, expected)
        aggregate.assert_called_once()
        left_records, right_records = aggregate.call_args.args[1:3]
        self.assertEqual(
            [record["partition"]["plan"] for record in left_records],
            ["coop", "no_coop"],
        )
        self.assertEqual(
            [record["partition"]["plan"] for record in right_records],
            ["coop", "no_coop"],
        )

    def test_exact_similarity_bridge_missing_description_fails_closed(self):
        scope = self._scope()
        left_target = {"course_id": 1, "program": "IT", "course_code": "00000001"}
        right_target = {"course_id": 2, "program": "IT", "course_code": "00000002"}
        left_request = EvidenceRequest(
            "left", "description_evidence", scope, course_targets=(left_target,)
        )
        right_request = EvidenceRequest(
            "right", "description_evidence", scope, course_targets=(right_target,)
        )
        bundle = EvidenceBundle(
            plan=EvidencePlan(scope=scope, requests=(left_request, right_request)),
            results=(
                EvidenceExecutionResult(
                    "left", "description_evidence", left_request, scope,
                    "insufficient_evidence", (), "description_missing"
                ),
                EvidenceExecutionResult(
                    "right", "description_evidence", right_request, scope,
                    "complete", ({
                        "chunk_id": "right",
                        "chunk_type": "description",
                        "course_id": 2,
                        "course_code": "00000002",
                        "program": "IT",
                        "text": "right",
                        "provenance": ({"source_page": 2},),
                    },),
                ),
            ),
        )
        with patch(
            "rag.evidence_executor.aggregate_exact_course_similarity",
            return_value=SimilarityEvidence(status="valid_empty"),
        ) as aggregate:
            result = execute_exact_similarity_from_bundle(
                DB_PATH, bundle, "left", "right"
            )

        self.assertEqual(result.status, "insufficient_evidence")
        aggregate.assert_called_once()

    def test_exact_similarity_bridge_malformed_evidence_fails_without_vector_access(self):
        scope = self._scope()
        bundle = self._similarity_bundle((scope,))
        result = bundle.results[0]
        malformed = EvidenceExecutionResult(
            result.request_id,
            result.kind,
            result.planned_request,
            result.effective_scope,
            "complete",
            ({"chunk_id": "", "chunk_type": "description"},),
        )
        malformed_bundle = EvidenceBundle(
            bundle.plan,
            (malformed, bundle.results[1]),
        )
        with patch("rag.evidence_executor.aggregate_exact_course_similarity") as aggregate:
            result = execute_exact_similarity_from_bundle(
                DB_PATH,
                malformed_bundle,
                "left_descriptions",
                "right_descriptions",
            )

        self.assertEqual(result.status, "insufficient_evidence")
        aggregate.assert_not_called()


if __name__ == "__main__":
    unittest.main()

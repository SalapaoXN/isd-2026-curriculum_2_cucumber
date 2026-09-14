import unittest
from dataclasses import FrozenInstanceError
import rag.evidence_planner as evidence_planner_module

from rag.evidence_planner import (
    APPLICABLE,
    EVIDENCE_PRIMITIVES,
    EvidencePlan,
    EvidenceRequest,
    StructuralScope,
    UNCONSTRAINED,
    build_structural_scope,
    plan_evidence,
)
from rag.query_spec import QuerySpec
from rag.resolution import CourseReferenceResolution, ResolutionOutcome


class EvidencePlannerModelTests(unittest.TestCase):
    def test_scope_distinguishes_expansion_from_unconstrained_axes(self):
        scope = StructuralScope(
            program="IT",
            expand_applicable=("plan",),
            unconstrained=("year", "semester"),
        )

        self.assertEqual(scope.expand_applicable, ("plan",))
        self.assertEqual(scope.unconstrained, ("year", "semester"))
        self.assertEqual(scope.plans, ())
        self.assertEqual(scope.years, ())
        self.assertEqual(scope.semesters, ())

    @staticmethod
    def _spec(**changes):
        values = {
            "original_question": "question",
            "normalized_question": "question",
            "program": "IT",
            "plans": (),
            "years": (),
            "semesters": (),
            "course_codes": (),
            "course_name": None,
            "category": None,
            "topic": None,
            "operations": ("list",),
            "group_by": (),
            "judgement": "none",
        }
        values.update(changes)
        return QuerySpec(**values)

    @staticmethod
    def _resolution(program="IT", candidates=(), resolved_plans=()):
        references = ()
        if candidates:
            references = (
                CourseReferenceResolution(
                    reference_type="course_code",
                    reference="06016420",
                    candidates=tuple(candidates),
                ),
            )
        return ResolutionOutcome("answer", (), program, references, resolved_plans)

    def test_context_resolved_plan_is_used_when_query_has_no_explicit_plan(self):
        scope = build_structural_scope(
            self._spec(),
            self._resolution(resolved_plans=("coop",)),
        )

        self.assertEqual(scope.plans, ("coop",))
        self.assertNotIn("plan", scope.expand_applicable)

    def test_nq_005_expands_only_applicable_plan_and_semester(self):
        scope = build_structural_scope(
            self._spec(group_by=("semester",), operations=("sum_credits", "compare")),
            self._resolution(),
        )

        self.assertEqual(scope.expand_applicable, ("plan", "semester"))
        self.assertEqual(scope.group_by, ("semester",))
        self.assertIn("year", scope.unconstrained)
        self.assertIn("course", scope.unconstrained)

    def test_nq_009_preserves_explicit_plan_order_without_expansion(self):
        scope = build_structural_scope(
            self._spec(plans=("coop", "no_coop"), group_by=("plan",)),
            self._resolution(),
        )

        self.assertEqual(scope.plans, ("coop", "no_coop"))
        self.assertNotIn("plan", scope.expand_applicable)
        self.assertEqual(scope.group_by, ("plan",))

    def test_nq_010_uses_resolved_program_and_preserves_exact_course(self):
        candidate = {
            "program": "IT",
            "course_code": "06016420",
            "course_id": 20,
        }
        scope = build_structural_scope(
            self._spec(
                program=None,
                course_codes=("06016420",),
                operations=("placement", "earliest", "compare"),
                group_by=("plan",),
            ),
            self._resolution(candidates=(candidate,)),
        )

        self.assertEqual(scope.program, "IT")
        self.assertEqual(scope.course_targets[0]["course_code"], "06016420")
        self.assertEqual(scope.expand_applicable, ("plan",))

    def test_nq_023_keeps_year_partitions_under_applicable_plans(self):
        scope = build_structural_scope(
            self._spec(years=(2, 3), group_by=("year",), topic="programming"),
            self._resolution(),
        )

        self.assertEqual(scope.years, (2, 3))
        self.assertEqual(scope.expand_applicable, ("plan",))
        self.assertEqual(scope.group_by, ("year",))

    def test_nq_030_preserves_one_exact_course_target_without_fabrication(self):
        candidate = {
            "program": "IT",
            "course_code": "06016420",
            "course_id": 20,
        }
        scope = build_structural_scope(
            self._spec(
                course_codes=("06016420",),
                operations=("describe", "placement"),
            ),
            self._resolution(candidates=(candidate,)),
        )

        self.assertEqual(len(scope.course_targets), 1)
        self.assertNotIn("course", scope.expand_applicable)

    def test_empty_axes_are_symbolic_and_no_cross_product_is_materialized(self):
        scope = build_structural_scope(self._spec(), self._resolution())

        self.assertEqual(scope.plans, ())
        self.assertIn("plan", scope.expand_applicable)
        self.assertEqual(scope.expand_applicable.count("plan"), 1)
        self.assertNotIn("partitions", scope.__slots__)

    def test_blocked_resolution_cannot_produce_a_scope(self):
        blocked = ResolutionOutcome("clarify_program", ("program",), None, ())

        with self.assertRaises(ValueError):
            build_structural_scope(self._spec(), blocked)

    def test_nq_005_maps_credit_comparison_to_partitioned_credit_evidence(self):
        spec = self._spec(
            operations=("sum_credits", "compare"),
            group_by=("semester",),
        )
        plan = plan_evidence(spec, self._resolution())

        self.assertEqual(plan.group_by, ("semester",))
        self.assertEqual(
            [request.kind for request in plan.requests],
            ["course_set", "credit_facts"],
        )
        self.assertEqual(plan.requests[1].depends_on, ("course_set",))
        self.assertIn("plan", plan.scope.expand_applicable)
        self.assertIn("semester", plan.scope.expand_applicable)
        self.assertEqual(plan.requests[0].scope, plan.requests[1].scope)

    def test_nq_009_creates_one_course_and_placement_request_per_plan(self):
        plan = plan_evidence(
            self._spec(
                plans=("coop", "no_coop"),
                operations=("compare",),
                group_by=("plan",),
            ),
            self._resolution(),
        )

        self.assertEqual(
            [(request.kind, request.scope.plans) for request in plan.requests],
            [
                ("course_set", ("coop",)),
                ("placement_facts", ("coop",)),
                ("course_set", ("no_coop",)),
                ("placement_facts", ("no_coop",)),
            ],
        )

    def test_nq_010_exact_course_placement_needs_no_global_topic_request(self):
        candidate = {"program": "IT", "course_code": "06016420", "course_id": 20}
        plan = plan_evidence(
            self._spec(
                program=None,
                course_codes=("06016420",),
                operations=("placement", "earliest", "compare"),
                group_by=("plan",),
            ),
            self._resolution(candidates=(candidate,)),
        )

        self.assertEqual(
            [request.kind for request in plan.requests], ["placement_facts"]
        )
        self.assertEqual(plan.requests[0].course_targets[0]["course_code"], "06016420")

    def test_nq_022_count_and_credits_share_one_topic_target(self):
        plan = plan_evidence(
            self._spec(
                topic="คอม",
                operations=("count", "sum_credits"),
            ),
            self._resolution(),
        )

        self.assertEqual(
            [request.kind for request in plan.requests],
            ["course_set", "topic_matches", "credit_facts"],
        )
        self.assertEqual(plan.requests[1].depends_on, ("course_set",))
        self.assertEqual(plan.requests[2].depends_on, ("topic_matches",))

    def test_nq_023_keeps_topic_target_under_plan_and_year_scope(self):
        plan = plan_evidence(
            self._spec(
                years=(2, 3),
                topic="programming",
                operations=("count", "compare"),
                group_by=("year",),
            ),
            self._resolution(),
        )

        self.assertEqual(
            [request.kind for request in plan.requests],
            ["course_set", "topic_matches"],
        )
        self.assertEqual(plan.scope.years, (2, 3))
        self.assertEqual(plan.scope.expand_applicable, ("plan",))
        self.assertEqual(plan.requests[1].depends_on, ("course_set",))

    def test_nq_017_constrains_topic_matches_to_one_course_set(self):
        plan = plan_evidence(
            self._spec(
                years=(2,),
                topic="database",
                operations=("list",),
            ),
            self._resolution(),
        )

        self.assertEqual(
            [request.kind for request in plan.requests],
            ["course_set", "topic_matches"],
        )
        self.assertEqual(plan.requests[1].topic, "database")
        self.assertEqual(plan.requests[1].depends_on, ("course_set",))
        self.assertEqual(plan.requests[0].scope, plan.requests[1].scope)

    def test_nq_036_reuses_full_structural_course_set_for_credits(self):
        plan = plan_evidence(
            self._spec(
                years=(3,),
                semesters=(2,),
                operations=("count", "sum_credits"),
            ),
            self._resolution(),
        )

        self.assertEqual(
            [request.kind for request in plan.requests],
            ["course_set", "credit_facts"],
        )
        self.assertEqual(plan.requests[1].depends_on, ("course_set",))
        self.assertEqual(plan.requests[0].scope, plan.requests[1].scope)
        self.assertEqual(plan.scope.years, (3,))
        self.assertEqual(plan.scope.semesters, (2,))

    def test_nq_030_and_nq_031_use_description_and_placement_shapes(self):
        candidates = (
            {"program": "IT", "course_code": "06016420", "course_id": 20},
            {"program": "IT", "course_code": "06016465", "course_id": 65},
        )
        resolution = ResolutionOutcome(
            "answer",
            (),
            "IT",
            (
                CourseReferenceResolution("course_code", "06016420", (candidates[0],)),
                CourseReferenceResolution("course_code", "06016465", (candidates[1],)),
            ),
        )
        describe_plan = plan_evidence(
            self._spec(
                course_codes=("06016420",),
                operations=("describe", "placement"),
            ),
            self._resolution(candidates=(candidates[0],)),
        )
        similarity_plan = plan_evidence(
            self._spec(
                course_codes=("06016420", "06016465"),
                operations=("similarity",),
                group_by=("course",),
            ),
            resolution,
        )

        self.assertEqual(
            [request.kind for request in describe_plan.requests],
            ["description_evidence", "placement_facts"],
        )
        self.assertEqual(
            [request.kind for request in similarity_plan.requests],
            ["description_evidence", "description_evidence"],
        )
        self.assertEqual(
            [len(request.course_targets) for request in similarity_plan.requests],
            [1, 1],
        )
        self.assertEqual(
            [request.scope.group_by for request in similarity_plan.requests],
            [("course",), ("course",)],
        )
        self.assertNotEqual(
            similarity_plan.requests[0].course_targets,
            similarity_plan.requests[1].course_targets,
        )
        self.assertEqual(
            describe_plan.requests[0].scope.course_targets,
            describe_plan.requests[1].scope.course_targets,
        )

    def test_every_request_requires_provenance_and_no_operation_adds_new_primitive(self):
        plan = plan_evidence(
            self._spec(
                operations=("list", "describe", "count", "sum_credits", "existence",
                            "compare", "earliest", "placement", "prerequisite",
                            "similarity"),
                course_codes=("06016420",),
            ),
            self._resolution(candidates=(
                {"program": "IT", "course_code": "06016420", "course_id": 20},
            )),
        )

        self.assertTrue(plan.requests)
        self.assertTrue(all(request.provenance_required for request in plan.requests))
        self.assertTrue(
            {request.kind for request in plan.requests}.issubset(set(EVIDENCE_PRIMITIVES))
        )

    def test_planner_mapping_has_no_runtime_execution_dependencies(self):
        self.assertNotIn("sqlite3", vars(evidence_planner_module))
        self.assertNotIn("retrieve", vars(evidence_planner_module))
        self.assertNotIn("model", vars(evidence_planner_module))
        self.assertNotIn("aggregate", vars(evidence_planner_module))

        plan = plan_evidence(self._spec(topic="database"), self._resolution())
        self.assertTrue(all(request.provenance_required for request in plan.requests))

    def test_plan_evidence_rejects_every_blocked_resolution(self):
        for action in ("unsupported", "no_data", "clarify_program"):
            with self.subTest(action=action):
                blocked = ResolutionOutcome(action, (), None, ())
                with self.assertRaises(ValueError):
                    plan_evidence(self._spec(), blocked)

    def test_scope_rejects_conflicting_axis_states(self):
        with self.assertRaises(ValueError):
            StructuralScope(
                expand_applicable=("year",),
                unconstrained=("year",),
            )

        with self.assertRaises(ValueError):
            StructuralScope(years=(2,), unconstrained=("year",))

    def test_request_accepts_only_the_six_primitives(self):
        scope = StructuralScope(program="IT")
        requests = tuple(
            EvidenceRequest(f"r{index}", kind, scope)
            for index, kind in enumerate(EVIDENCE_PRIMITIVES, start=1)
        )

        self.assertEqual(tuple(request.kind for request in requests), EVIDENCE_PRIMITIVES)
        with self.assertRaises(ValueError):
            EvidenceRequest("bad", "arithmetic", scope)

    def test_request_preserves_explicit_targets_and_dependencies_immutably(self):
        scope = StructuralScope(program="IT")
        target = {
            "program": "IT",
            "course_code": "06016420",
            "identity": {"course_id": 20},
        }
        request = EvidenceRequest(
            "description",
            "description_evidence",
            scope,
            depends_on=("course_set",),
            course_targets=(target,),
        )

        self.assertEqual(request.depends_on, ("course_set",))
        self.assertEqual(request.course_targets[0]["course_code"], "06016420")
        with self.assertRaises(TypeError):
            request.course_targets[0]["course_code"] = "06016421"
        with self.assertRaises(FrozenInstanceError):
            request.kind = "course_set"

    def test_plan_preserves_grouping_and_checks_dependencies(self):
        scope = StructuralScope(
            program="IT",
            expand_applicable=("plan",),
            unconstrained=("semester",),
        )
        course_set = EvidenceRequest("course_set", "course_set", scope)
        topics = EvidenceRequest(
            "topics",
            "topic_matches",
            scope,
            depends_on=("course_set",),
            topic="database",
        )
        plan = EvidencePlan(scope, (course_set, topics), ("semester",))

        self.assertEqual(plan.group_by, ("semester",))
        self.assertEqual(plan.requests[1].depends_on, ("course_set",))

        with self.assertRaises(ValueError):
            EvidencePlan(scope, (topics,), ())

    def test_representations_are_frozen(self):
        scope = StructuralScope(program="IT")
        request = EvidenceRequest("courses", "course_set", scope)
        plan = EvidencePlan(scope, (request,))

        with self.assertRaises(FrozenInstanceError):
            scope.program = "BIT"
        with self.assertRaises(FrozenInstanceError):
            request.scope = StructuralScope(program="BIT")
        with self.assertRaises(FrozenInstanceError):
            plan.requests = ()


if __name__ == "__main__":
    unittest.main()

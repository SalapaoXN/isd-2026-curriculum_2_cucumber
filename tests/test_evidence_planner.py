import unittest

from rag.evidence_planner import plan_evidence
from rag.query_spec import QuerySpec
from rag.resolution import CourseReferenceResolution, ResolutionOutcome


class TopicPrerequisitePlannerTests(unittest.TestCase):
    @staticmethod
    def _spec(**changes):
        values = {
            "original_question": "question",
            "normalized_question": "question",
            "program": "IT",
            "plans": (),
            "years": (3,),
            "semesters": (),
            "course_codes": (),
            "course_name": None,
            "category": None,
            "topic": "data",
            "operations": ("list",),
            "group_by": (),
            "judgement": "none",
        }
        values.update(changes)
        return QuerySpec(**values)

    @staticmethod
    def _resolution(candidates=()):
        references = ()
        if candidates:
            references = (
                CourseReferenceResolution(
                    reference_type="course_code",
                    reference="06016420",
                    candidates=tuple(candidates),
                ),
            )
        return ResolutionOutcome("answer", (), "IT", references, ())

    def test_topic_list_without_prerequisite_request_is_unchanged(self):
        plan = plan_evidence(self._spec(), self._resolution())

        self.assertEqual(
            [(request.request_id, request.kind) for request in plan.requests],
            [("course_set", "course_set"), ("topic_matches", "topic_matches")],
        )

    def test_topic_list_prerequisite_adds_one_dependency_request(self):
        plan = plan_evidence(
            self._spec(operations=("list", "prerequisite")),
            self._resolution(),
        )

        self.assertEqual(
            [(request.request_id, request.kind) for request in plan.requests],
            [
                ("course_set", "course_set"),
                ("topic_matches", "topic_matches"),
                ("topic_prerequisite_facts", "prerequisite_facts"),
            ],
        )
        prerequisite = plan.requests[-1]
        self.assertEqual(prerequisite.depends_on, ("topic_matches",))
        self.assertEqual(prerequisite.course_targets, ())
        self.assertEqual(prerequisite.scope.program, "IT")
        self.assertEqual(prerequisite.scope.years, (3,))

    def test_exact_course_prerequisite_request_remains_direct(self):
        target = {"program": "IT", "course_code": "06016420", "course_id": 20}
        plan = plan_evidence(
            self._spec(
                topic=None,
                operations=("prerequisite",),
                course_codes=("06016420",),
            ),
            self._resolution(candidates=(target,)),
        )

        self.assertEqual(len(plan.requests), 1)
        request = plan.requests[0]
        self.assertEqual(request.request_id, "prerequisite_facts_1")
        self.assertEqual(request.depends_on, ())
        self.assertEqual(request.course_targets, (target,))

    def test_topic_prerequisite_does_not_widen_structural_scope(self):
        plan = plan_evidence(
            self._spec(
                plans=("coop",),
                years=(3,),
                semesters=(1,),
                operations=("list", "prerequisite"),
            ),
            self._resolution(),
        )

        request = plan.requests[-1]
        self.assertEqual(request.scope.program, "IT")
        self.assertEqual(request.scope.plans, ("coop",))
        self.assertEqual(request.scope.years, (3,))
        self.assertEqual(request.scope.semesters, (1,))
        self.assertEqual(request.scope.course_targets, ())


if __name__ == "__main__":
    unittest.main()

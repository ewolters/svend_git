"""Behavioral tests for Harada Method API — TST-001 §10.5-10.6.

Tests exercise real user workflows: take questionnaire, set goals,
manage 64-window, track routines, write diary entries.

No existence-only tests per TST-001 §10.6.
<!-- test: agents_api.tests.test_harada -->
"""

import json
import uuid
from datetime import date

from django.test import Client, TestCase, override_settings

from accounts.models import Tier, User
from core.models import (
    ArchetypeAssignment,
    DailyDiary,
    QuestionDimension,
    QuestionnaireResponse,
    RoutineCheck,
    Scenario,
    Window64,
)

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email, tier=Tier.PRO, password="testpass123!"):
    username = email.split("@")[0]
    user = User.objects.create_user(username=username, email=email, password=password)
    user.tier = tier
    user.is_email_verified = True
    user.save(update_fields=["tier", "is_email_verified"])
    return user


def _authed_client(user):
    client = Client()
    client.force_login(user)
    return client


def _post_json(client, url, data):
    return client.post(url, json.dumps(data), content_type="application/json")


def _patch_json(client, url, data):
    return client.patch(url, json.dumps(data), content_type="application/json")


def _seed_ci_readiness():
    """Create minimal CI Readiness dimensions + scenarios for testing."""
    # One Likert dimension
    likert_dim = QuestionDimension.objects.create(
        instrument="ci_readiness",
        dimension_number=3,
        name="Early Win Dependency",
        category="ci",
        response_type="likert",
        question_text="I can sustain effort on a project for months without external validation.",
    )

    # One forced-choice dimension with 2 scenarios
    fc_dim = QuestionDimension.objects.create(
        instrument="ci_readiness",
        dimension_number=1,
        name="Process vs. Person Attribution",
        category="ci",
        response_type="forced_choice",
        question_text="When something goes wrong, where does your attention go first?",
    )
    s1 = Scenario.objects.create(
        dimension=fc_dim,
        scenario_key="d1_s1",
        situation="A defect rate spikes after a shift change.",
        option_a="Talk to the shift lead.",
        option_a_label="person_focused",
        option_b="Pull the data.",
        option_b_label="data_first",
        option_c="Walk the process.",
        option_c_label="system_thinker",
        option_d="Check patterns.",
        option_d_label="pattern_seeker",
    )
    s2 = Scenario.objects.create(
        dimension=fc_dim,
        scenario_key="d1_s2",
        situation="Customer complaint about dimensional variation.",
        option_a="Ask who was running the machine.",
        option_a_label="person_focused",
        option_b="Pull SPC charts.",
        option_b_label="data_first",
        option_c="Check maintenance log.",
        option_c_label="system_thinker",
        option_d="Look at other parts on same equipment.",
        option_d_label="pattern_seeker",
    )

    return likert_dim, fc_dim, s1, s2


# ===========================================================================
# Questionnaire
# ===========================================================================


@SECURE_OFF
class QuestionnaireFlowTest(TestCase):
    """
    Full questionnaire workflow: present → submit → history.

    <!-- test: agents_api.tests.test_harada.QuestionnaireFlowTest -->
    """

    def setUp(self):
        self.user = _make_user("harada@test.com")
        self.client = _authed_client(self.user)
        self.likert_dim, self.fc_dim, self.s1, self.s2 = _seed_ci_readiness()

    def test_get_questionnaire_returns_dimensions(self):
        """GET /api/harada/questionnaire/ returns Likert and forced-choice items."""
        res = self.client.get("/api/harada/questionnaire/?instrument=ci_readiness")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["dimensions"]), 2)
        self.assertIn("session_id", data)

        # Check Likert dimension
        likert = next(d for d in data["dimensions"] if d["response_type"] == "likert")
        self.assertEqual(likert["scale_min"], 1)
        self.assertEqual(likert["scale_max"], 5)
        self.assertIn("question_text", likert)

        # Check forced-choice dimension
        fc = next(d for d in data["dimensions"] if d["response_type"] == "forced_choice")
        self.assertIn("scenario_id", fc)
        self.assertIn("situation", fc)
        self.assertEqual(len(fc["options"]), 4)

    def test_options_are_randomized(self):
        """Forced-choice options should not always be in the same order."""
        orders = set()
        for _ in range(20):
            res = self.client.get("/api/harada/questionnaire/?instrument=ci_readiness")
            fc = next(d for d in res.json()["dimensions"] if d["response_type"] == "forced_choice")
            order = tuple(o["label"] for o in fc["options"])
            orders.add(order)
        # With 4! = 24 permutations and 20 draws, should see at least 2 different orders
        self.assertGreater(len(orders), 1, "Options should be randomized across requests")

    def test_submit_and_retrieve_responses(self):
        """Submit responses and verify they are stored and retrievable."""
        # Get questionnaire to get IDs
        res = self.client.get("/api/harada/questionnaire/?instrument=ci_readiness")
        data = res.json()
        session_id = data["session_id"]
        dims = data["dimensions"]

        likert = next(d for d in dims if d["response_type"] == "likert")
        fc = next(d for d in dims if d["response_type"] == "forced_choice")

        # Submit
        res = _post_json(
            self.client,
            "/api/harada/questionnaire/submit/",
            {
                "instrument": "ci_readiness",
                "session_id": session_id,
                "responses": [
                    {"dimension_id": likert["dimension_id"], "score": 4},
                    {
                        "dimension_id": fc["dimension_id"],
                        "scenario_id": fc["scenario_id"],
                        "option_chosen": "system_thinker",
                    },
                ],
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["responses_saved"], 2)
        self.assertEqual(res.json()["version"], 1)

        # Verify in DB
        self.assertEqual(QuestionnaireResponse.objects.filter(user=self.user).count(), 2)

        # Retrieve history
        res = self.client.get("/api/harada/questionnaire/history/?instrument=ci_readiness")
        self.assertEqual(res.status_code, 200)
        history = res.json()["history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["version"], 1)
        self.assertEqual(len(history[0]["responses"]), 2)

    def test_retake_increments_version(self):
        """Second submission gets version 2."""
        # First take
        _post_json(
            self.client,
            "/api/harada/questionnaire/submit/",
            {
                "instrument": "ci_readiness",
                "session_id": "00000000-0000-0000-0000-000000000001",
                "responses": [{"dimension_id": str(self.likert_dim.id), "score": 3}],
            },
        )

        # Second take
        res = _post_json(
            self.client,
            "/api/harada/questionnaire/submit/",
            {
                "instrument": "ci_readiness",
                "session_id": "00000000-0000-0000-0000-000000000002",
                "responses": [{"dimension_id": str(self.likert_dim.id), "score": 5}],
            },
        )
        self.assertEqual(res.json()["version"], 2)

    def test_q11_experience_gate_uses_profile(self):
        """Q11 shows experienced variant when user has experience_level set."""
        # Set user as intermediate
        self.user.experience_level = "intermediate"
        self.user.save(update_fields=["experience_level"])

        # Add Q11 dimension
        QuestionDimension.objects.create(
            instrument="ci_readiness",
            dimension_number=11,
            name="Measurement System Trust",
            category="ci",
            response_type="likert",
            question_text="I have delayed or revised a conclusion after discovering a problem with how the data was collected.",
        )

        res = self.client.get("/api/harada/questionnaire/?instrument=ci_readiness")
        data = res.json()
        d11 = next((d for d in data["dimensions"] if d["dimension_number"] == 11), None)
        self.assertIsNotNone(d11)
        self.assertIn("delayed or revised", d11["question_text"])
        self.assertNotIn("experience_question", data)

    def test_q11_experience_gate_early_career(self):
        """Q11 shows early-career variant for students."""
        self.user.role = "student"
        self.user.save(update_fields=["role"])

        QuestionDimension.objects.create(
            instrument="ci_readiness",
            dimension_number=11,
            name="Measurement System Trust",
            category="ci",
            response_type="likert",
            question_text="I have delayed or revised a conclusion after discovering a problem with how the data was collected.",
        )

        res = self.client.get("/api/harada/questionnaire/?instrument=ci_readiness")
        d11 = next((d for d in res.json()["dimensions"] if d["dimension_number"] == 11), None)
        self.assertIn("investigate whether the measurement system", d11["question_text"])

    def test_q11_experience_gate_asks_when_unknown(self):
        """When experience not in profile, response includes experience_question."""
        self.user.experience_level = ""
        self.user.role = ""
        self.user.save(update_fields=["experience_level", "role"])

        QuestionDimension.objects.create(
            instrument="ci_readiness",
            dimension_number=11,
            name="Measurement System Trust",
            category="ci",
            response_type="likert",
            question_text="I have delayed or revised a conclusion after discovering a problem with how the data was collected.",
        )

        res = self.client.get("/api/harada/questionnaire/?instrument=ci_readiness")
        data = res.json()
        self.assertIn("experience_question", data)
        self.assertEqual(len(data["experience_question"]["options"]), 2)

    def test_experience_answer_stored_on_submit(self):
        """Submitting experience_answer updates user profile."""
        self.user.experience_level = ""
        self.user.save(update_fields=["experience_level"])

        _post_json(
            self.client,
            "/api/harada/questionnaire/submit/",
            {
                "instrument": "ci_readiness",
                "session_id": "00000000-0000-0000-0000-000000000099",
                "responses": [{"dimension_id": str(self.likert_dim.id), "score": 3}],
                "experience_answer": "early",
            },
        )

        self.user.refresh_from_db()
        self.assertEqual(self.user.experience_level, "beginner")

    def test_retake_avoids_previous_scenario(self):
        """On retake, previously seen scenarios are avoided when alternatives exist."""
        # First take — record which scenario was used
        res = self.client.get("/api/harada/questionnaire/?instrument=ci_readiness")
        fc = next(d for d in res.json()["dimensions"] if d["response_type"] == "forced_choice")
        first_scenario = fc["scenario_id"]

        # Submit first take
        _post_json(
            self.client,
            "/api/harada/questionnaire/submit/",
            {
                "instrument": "ci_readiness",
                "session_id": res.json()["session_id"],
                "responses": [
                    {
                        "dimension_id": fc["dimension_id"],
                        "scenario_id": first_scenario,
                        "option_chosen": "data_first",
                    },
                ],
            },
        )

        # Retake — should prefer the other scenario
        seen_on_retake = set()
        for _ in range(10):
            res = self.client.get("/api/harada/questionnaire/?instrument=ci_readiness")
            fc = next(d for d in res.json()["dimensions"] if d["response_type"] == "forced_choice")
            seen_on_retake.add(fc["scenario_id"])

        # Should have seen the OTHER scenario at least once
        self.assertGreater(len(seen_on_retake), 0)
        # With 2 scenarios and preference for unseen, the other should appear
        other_scenarios = seen_on_retake - {first_scenario}
        self.assertTrue(len(other_scenarios) > 0, "Retake should prefer unseen scenarios")


# ===========================================================================
# Goals
# ===========================================================================


@SECURE_OFF
class GoalCascadeTest(TestCase):
    """
    Goal cascade: long-term → FY → month → immediate.

    <!-- test: agents_api.tests.test_harada.GoalCascadeTest -->
    """

    def setUp(self):
        self.user = _make_user("goals@test.com")
        self.client = _authed_client(self.user)

    def test_create_goal_cascade(self):
        """Create long-term → FY → month hierarchy."""
        # Long-term
        res = _post_json(
            self.client,
            "/api/harada/goals/",
            {
                "title": "Build Svend to $4500 MRR",
                "horizon": "long_term",
                "service_at_work": "Accessible CI tools for emerging markets",
            },
        )
        self.assertEqual(res.status_code, 201)
        lt_id = res.json()["id"]

        # FY under long-term
        res = _post_json(
            self.client,
            "/api/harada/goals/",
            {
                "title": "45 paying users by September",
                "horizon": "fiscal_year",
                "parent_id": lt_id,
            },
        )
        self.assertEqual(res.status_code, 201)

        # List — should show tree
        res = self.client.get("/api/harada/goals/")
        goals = res.json()["goals"]
        self.assertEqual(len(goals), 1)  # One root
        self.assertEqual(len(goals[0]["children"]), 1)  # One child

    def test_achieve_goal(self):
        """Marking a goal achieved sets achieved_at."""
        res = _post_json(
            self.client,
            "/api/harada/goals/",
            {"title": "First ILSSI cohort", "horizon": "immediate"},
        )
        goal_id = res.json()["id"]

        res = _patch_json(
            self.client,
            f"/api/harada/goals/{goal_id}/",
            {"status": "achieved"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "achieved")
        self.assertIsNotNone(res.json()["achieved_at"])


# ===========================================================================
# 64-Window
# ===========================================================================


@SECURE_OFF
class Window64Test(TestCase):
    """
    64-window: 8 goals × 8 actions.

    <!-- test: agents_api.tests.test_harada.Window64Test -->
    """

    def setUp(self):
        self.user = _make_user("window@test.com")
        self.client = _authed_client(self.user)

    def test_create_goal_and_actions(self):
        """Create a center goal and surrounding task/routine cells."""
        # Center goal
        res = _post_json(
            self.client,
            "/api/harada/window/",
            {"goal_number": 1, "position": 0, "text": "Build ILSSI pipeline"},
        )
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["cell_type"], "goal")

        # Routine action
        res = _post_json(
            self.client,
            "/api/harada/window/",
            {
                "goal_number": 1,
                "position": 1,
                "text": "Send 5 outreach emails",
                "cell_type": "routine",
            },
        )
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["cell_type"], "routine")

        # Task action
        res = _post_json(
            self.client,
            "/api/harada/window/",
            {
                "goal_number": 1,
                "position": 2,
                "text": "Set up student batch enrollment",
                "cell_type": "task",
            },
        )
        self.assertEqual(res.status_code, 201)

        # Get grid
        res = self.client.get("/api/harada/window/")
        grid = res.json()["window"]
        self.assertIn("1", grid)
        self.assertEqual(len(grid["1"]), 3)

    def test_complete_task(self):
        """Mark a task as completed."""
        _post_json(
            self.client,
            "/api/harada/window/",
            {
                "goal_number": 1,
                "position": 3,
                "text": "Write standard",
                "cell_type": "task",
            },
        )
        cell = Window64.objects.get(user=self.user, goal_number=1, position=3)

        res = _patch_json(
            self.client,
            f"/api/harada/window/{cell.id}/",
            {"is_completed": True},
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["is_completed"])


# ===========================================================================
# Routine Tracker
# ===========================================================================


@SECURE_OFF
class RoutineTrackerTest(TestCase):
    """
    Daily routine tracking with streak calculation.

    <!-- test: agents_api.tests.test_harada.RoutineTrackerTest -->
    """

    def setUp(self):
        self.user = _make_user("routine@test.com")
        self.client = _authed_client(self.user)
        # Create a routine
        self.routine = Window64.objects.create(
            user=self.user,
            goal_number=1,
            position=1,
            cell_type="routine",
            text="5 emails daily",
        )

    def test_check_routine(self):
        """Check a routine for today."""
        res = _post_json(
            self.client,
            "/api/harada/routines/",
            {"window_cell_id": str(self.routine.id), "is_completed": True},
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["is_completed"])

        # Verify in DB
        self.assertTrue(RoutineCheck.objects.filter(user=self.user, date=date.today()).exists())

    def test_get_daily_checklist(self):
        """GET returns today's routines with completion status."""
        res = self.client.get("/api/harada/routines/")
        self.assertEqual(res.status_code, 200)
        routines = res.json()["routines"]
        self.assertEqual(len(routines), 1)
        self.assertFalse(routines[0]["is_completed"])

    def test_check_is_idempotent(self):
        """Checking same routine twice on same day updates, doesn't duplicate."""
        _post_json(
            self.client,
            "/api/harada/routines/",
            {"window_cell_id": str(self.routine.id), "is_completed": True},
        )
        _post_json(
            self.client,
            "/api/harada/routines/",
            {"window_cell_id": str(self.routine.id), "is_completed": False},
        )
        self.assertEqual(RoutineCheck.objects.filter(user=self.user, date=date.today()).count(), 1)
        check = RoutineCheck.objects.get(user=self.user, date=date.today())
        self.assertFalse(check.is_completed)  # Updated to false


# ===========================================================================
# Daily Diary
# ===========================================================================


@SECURE_OFF
class DailyDiaryTest(TestCase):
    """
    Daily diary with 8-dimension scoring and reflection.

    <!-- test: agents_api.tests.test_harada.DailyDiaryTest -->
    """

    def setUp(self):
        self.user = _make_user("diary@test.com")
        self.client = _authed_client(self.user)

    def test_create_diary_entry(self):
        """Create today's diary with scores and tasks."""
        res = _post_json(
            self.client,
            "/api/harada/diary/",
            {
                "daily_phrase": "Ship it",
                "scores": {
                    "overall": 4,
                    "mental": 3,
                    "body": 3,
                    "work": 5,
                    "relations": 3,
                    "life": 3,
                    "learning": 5,
                    "routines": 4,
                },
                "top_tasks": [
                    {"task": "Build Harada API", "completed": True},
                    {"task": "Draft CI questions", "completed": True},
                    {"task": "Steam room", "completed": False},
                ],
                "challenges": "ILSSI partnership is real.",
                "what_differently": "Started training center infra earlier.",
                "score_comments": {"mental": "Recovered", "work": "Massive progress"},
            },
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["score_total"], 30)
        self.assertEqual(data["tasks_completed"], 2)
        self.assertEqual(data["scores"]["work"], 5)

    def test_update_diary_preserves_data(self):
        """PATCH updates specific fields without losing others."""
        _post_json(
            self.client,
            "/api/harada/diary/",
            {
                "daily_phrase": "Morning",
                "scores": {"overall": 3, "mental": 2},
            },
        )

        res = _patch_json(
            self.client,
            f"/api/harada/diary/{date.today()}/",
            {
                "scores": {"mental": 4},
                "challenges": "Feeling better after steam room",
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["scores"]["mental"], 4)
        self.assertEqual(res.json()["scores"]["overall"], 3)  # Preserved
        self.assertEqual(res.json()["challenges"], "Feeling better after steam room")

    def test_one_entry_per_day(self):
        """Second POST to same date updates, doesn't duplicate."""
        _post_json(self.client, "/api/harada/diary/", {"daily_phrase": "v1"})
        _post_json(self.client, "/api/harada/diary/", {"daily_phrase": "v2"})
        self.assertEqual(DailyDiary.objects.filter(user=self.user, date=date.today()).count(), 1)
        self.assertEqual(DailyDiary.objects.get(user=self.user, date=date.today()).daily_phrase, "v2")

    def test_list_recent_entries(self):
        """GET returns recent diary entries."""
        DailyDiary.objects.create(user=self.user, date=date.today(), daily_phrase="Today")
        res = self.client.get("/api/harada/diary/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["entries"]), 1)


# ===========================================================================
# Clustering Pipeline
# ===========================================================================


@SECURE_OFF
class ClusteringPipelineTest(TestCase):
    """
    K-prototypes clustering on CI Readiness responses.

    Uses synthetic data to validate the pipeline since real user data
    requires 10+ complete responses.
    <!-- test: agents_api.tests.test_harada.ClusteringPipelineTest -->
    """

    def _seed_dimensions(self):
        """Create all 12 CI Readiness dimensions."""
        dims = {}
        for n in [3, 4, 5, 6, 8, 10, 11]:
            dims[n] = QuestionDimension.objects.create(
                instrument="ci_readiness",
                dimension_number=n,
                name=f"Likert Dim {n}",
                category="ci",
                response_type="likert",
                question_text=f"Test question {n}",
            )
        for n in [1, 2, 7, 9, 12]:
            dims[n] = QuestionDimension.objects.create(
                instrument="ci_readiness",
                dimension_number=n,
                name=f"FC Dim {n}",
                category="ci",
                response_type="forced_choice",
                question_text=f"Test scenario {n}",
            )
        return dims

    def _create_synthetic_user(self, email, dims, likert_scores, fc_choices):
        """Create a user with complete CI Readiness responses."""
        user = _make_user(email)
        session = uuid.uuid4()

        for dim_num, score in zip([3, 4, 5, 6, 8, 10, 11], likert_scores):
            QuestionnaireResponse.objects.create(
                user=user,
                dimension=dims[dim_num],
                session_id=session,
                score=score,
            )

        for dim_num, choice in zip([1, 2, 7, 9, 12], fc_choices):
            QuestionnaireResponse.objects.create(
                user=user,
                dimension=dims[dim_num],
                session_id=session,
                option_chosen=choice,
            )

        return user

    def test_collect_response_matrix(self):
        """collect_response_matrix builds correct feature matrices."""
        from agents_api.clustering import collect_response_matrix

        dims = self._seed_dimensions()
        self._create_synthetic_user(
            "u1@test.com",
            dims,
            [5, 4, 3, 4, 5, 4, 3],
            ["system_thinker", "data_purist", "gemba_first", "coach", "fatigue_aware"],
        )

        users, likert, cat, fvs = collect_response_matrix()
        self.assertEqual(len(users), 1)
        self.assertEqual(likert.shape, (1, 7))
        self.assertEqual(cat.shape, (1, 5))
        self.assertEqual(likert[0, 0], 5.0)  # First Likert score
        self.assertEqual(cat[0, 0], "system_thinker")  # First FC choice

    def test_clustering_skips_insufficient_users(self):
        """Clustering returns skip when fewer than MIN_USERS_FOR_CLUSTERING."""
        from agents_api.clustering import run_clustering

        result = run_clustering()
        self.assertTrue(result["skipped"])
        self.assertIn("insufficient_users", result["reason"])

    def test_clustering_runs_with_enough_users(self):
        """Clustering produces assignments with 10+ synthetic users."""
        from agents_api.clustering import run_clustering

        dims = self._seed_dimensions()

        # Create two distinct "archetypes" in synthetic data
        for i in range(6):
            # Archetype A: high likert, system thinkers
            self._create_synthetic_user(
                f"archA_{i}@test.com",
                dims,
                [5, 5, 4, 5, 5, 4, 5],
                [
                    "system_thinker",
                    "nuanced_thinker",
                    "prepare_thoroughly",
                    "coach",
                    "fatigue_aware",
                ],
            )

        for i in range(6):
            # Archetype B: lower likert, person-focused fixers
            self._create_synthetic_user(
                f"archB_{i}@test.com",
                dims,
                [2, 2, 3, 2, 2, 3, 2],
                ["person_focused", "data_purist", "momentum_first", "fixer", "driver"],
            )

        result = run_clustering()
        self.assertFalse(result.get("skipped", False))
        self.assertIn("k", result)
        self.assertEqual(result["users"], 12)
        self.assertEqual(result["assignments"], 12)

        # Verify assignments stored
        total = ArchetypeAssignment.objects.count()
        self.assertEqual(total, 12)

        # Check that the two archetypes got different clusters
        a_clusters = set(
            ArchetypeAssignment.objects.filter(user__email__startswith="archA_").values_list("cluster_id", flat=True)
        )
        b_clusters = set(
            ArchetypeAssignment.objects.filter(user__email__startswith="archB_").values_list("cluster_id", flat=True)
        )
        # With clear separation, they should be in different clusters
        self.assertNotEqual(a_clusters, b_clusters, "Distinct archetypes should cluster separately")

    def test_archetype_api(self):
        """GET /api/harada/archetype/ returns assignment."""
        user = _make_user("arch_api@test.com")
        client = _authed_client(user)

        # No assignment yet
        res = client.get("/api/harada/archetype/")
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(res.json()["archetype"])

        # Create assignment
        ArchetypeAssignment.objects.create(
            user=user,
            session_id=uuid.uuid4(),
            instrument_version=1,
            cluster_id=0,
            cluster_label="System Thinker",
            feature_vector={"likert": {}, "categorical": {}},
        )

        res = client.get("/api/harada/archetype/")
        data = res.json()
        self.assertIsNotNone(data["archetype"])
        self.assertEqual(data["archetype"]["cluster_label"], "System Thinker")
        self.assertEqual(len(data["trajectory"]), 1)

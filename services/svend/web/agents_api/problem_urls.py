"""Problem Session API URLs."""

from django.urls import path

from . import problem_views as views

urlpatterns = [
    # List and create
    path("", views.problems_list, name="problems_list"),

    # Detail operations
    path("<uuid:problem_id>/", views.problem_detail, name="problem_detail"),

    # Hypothesis operations
    path("<uuid:problem_id>/hypotheses/", views.add_hypothesis, name="add_hypothesis"),
    path("<uuid:problem_id>/hypotheses/generate/", views.generate_hypotheses, name="generate_hypotheses"),
    path("<uuid:problem_id>/hypotheses/<str:hypothesis_id>/reject/", views.reject_hypothesis, name="reject_hypothesis"),

    # Evidence operations
    path("<uuid:problem_id>/evidence/", views.add_evidence, name="add_evidence"),

    # Resolution
    path("<uuid:problem_id>/resolve/", views.resolve_problem, name="resolve_problem"),

    # Interview (Decision Guide)
    path("<uuid:problem_id>/interview/start/", views.start_interview, name="start_interview"),
    path("<uuid:problem_id>/interview/answer/", views.interview_answer, name="interview_answer"),
    path("<uuid:problem_id>/interview/skip/", views.interview_skip, name="interview_skip"),
    path("<uuid:problem_id>/interview/save/", views.interview_save, name="interview_save"),
    path("<uuid:problem_id>/interview/status/", views.interview_status, name="interview_status"),

    # Methodology & Phase Management
    path("<uuid:problem_id>/methodology/", views.set_methodology, name="set_methodology"),
    path("<uuid:problem_id>/phase/advance/", views.advance_phase, name="advance_phase"),
    path("<uuid:problem_id>/phase/guidance/", views.get_phase_guidance, name="get_phase_guidance"),

    # Context file
    path("<uuid:problem_id>/context/", views.get_context_file, name="get_context_file"),
]

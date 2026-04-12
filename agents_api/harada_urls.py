"""URL routes for Harada Method API."""

from django.urls import path

from . import harada_views

urlpatterns = [
    # Questionnaire
    path("questionnaire/", harada_views.get_questionnaire, name="harada_questionnaire"),
    path("questionnaire/submit/", harada_views.submit_responses, name="harada_submit"),
    path(
        "questionnaire/history/",
        harada_views.get_response_history,
        name="harada_history",
    ),
    # Archetype
    path("archetype/", harada_views.get_archetype, name="harada_archetype"),
    # Goals
    path("goals/", harada_views.list_create_goals, name="harada_goals"),
    path("goals/<uuid:goal_id>/", harada_views.goal_detail, name="harada_goal_detail"),
    # 64-Window
    path("window/", harada_views.list_create_window, name="harada_window"),
    path(
        "window/<uuid:cell_id>/",
        harada_views.update_window_cell,
        name="harada_window_cell",
    ),
    # Routines
    path("routines/", harada_views.check_routine, name="harada_routines"),
    path("routines/history/", harada_views.routine_history, name="harada_routine_history"),
    # Daily Diary
    path("diary/", harada_views.list_create_diary, name="harada_diary"),
    path("diary/<str:diary_date>/", harada_views.diary_detail, name="harada_diary_detail"),
]

"""URL routes for Harada Method API."""

from django.urls import path

from . import views

urlpatterns = [
    # Questionnaire
    path("questionnaire/", views.get_questionnaire, name="harada_questionnaire"),
    path("questionnaire/submit/", views.submit_responses, name="harada_submit"),
    path(
        "questionnaire/history/",
        views.get_response_history,
        name="harada_history",
    ),
    # Archetype
    path("archetype/", views.get_archetype, name="harada_archetype"),
    # Goals
    path("goals/", views.list_create_goals, name="harada_goals"),
    path("goals/<uuid:goal_id>/", views.goal_detail, name="harada_goal_detail"),
    # 64-Window
    path("window/", views.list_create_window, name="harada_window"),
    path(
        "window/<uuid:cell_id>/",
        views.update_window_cell,
        name="harada_window_cell",
    ),
    # Routines
    path("routines/", views.check_routine, name="harada_routines"),
    path("routines/history/", views.routine_history, name="harada_routine_history"),
    # Daily Diary
    path("diary/", views.list_create_diary, name="harada_diary"),
    path("diary/<str:diary_date>/", views.diary_detail, name="harada_diary_detail"),
]

"""PCL — Process Characteristics Library URL routes."""

from django.urls import path

from . import views

app_name = "pcl"

urlpatterns = [
    path("measures/", views.measure_list, name="measure-list"),
    path("measures/create/", views.measure_create, name="measure-create"),
    path("measures/search/", views.measure_search, name="measure-search"),
    path("measures/<uuid:measure_id>/", views.measure_detail, name="measure-detail"),
    path("read/<slug:slug>/", views.measure_read, name="measure-read"),
    path("write/", views.datapoint_write, name="datapoint-write"),
    path("target/", views.target_set, name="target-set"),
]

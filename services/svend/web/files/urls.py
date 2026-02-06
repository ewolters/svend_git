"""File management URL configuration."""

from django.urls import path

from . import views

app_name = "files"

urlpatterns = [
    # File listing and upload
    path("", views.list_files, name="list"),
    path("upload/", views.upload_file, name="upload"),

    # Individual file operations
    path("<uuid:file_id>/", views.file_detail, name="detail"),
    path("<uuid:file_id>/download/", views.download_file, name="download"),

    # Sharing
    path("<uuid:file_id>/share/", views.create_share_link, name="share"),
    path("<uuid:file_id>/unshare/", views.revoke_share_link, name="unshare"),
    path("shared/<str:share_token>/", views.shared_file, name="shared"),

    # Quota and folders
    path("quota/", views.storage_quota, name="quota"),
    path("folders/", views.list_folders, name="folders"),
]

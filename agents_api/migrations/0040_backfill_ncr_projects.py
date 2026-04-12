"""Backfill core.Project for orphan NCRs.

Creates an auto-project for every NCR that has no project FK set.
Tags each auto-created project with ["auto-created", "ncr"].

Reversible: the reverse operation nulls out the project FK on NCRs
whose project has both auto-created tags (does NOT delete the projects
to avoid data loss).
"""

from django.db import migrations


def backfill_ncr_projects(apps, schema_editor):
    NonconformanceRecord = apps.get_model("agents_api", "NonconformanceRecord")
    Project = apps.get_model("core", "Project")

    orphans = NonconformanceRecord.objects.filter(project__isnull=True)
    count = 0
    for ncr in orphans.iterator():
        project = Project.objects.create(
            user_id=ncr.owner_id,
            title=ncr.title or "NCR Investigation",
            methodology="none",
            tags=["auto-created", "ncr"],
        )
        ncr.project = project
        ncr.save(update_fields=["project"])
        count += 1

    if count:
        print(f"\n  Backfilled {count} NCR(s) with auto-created projects.")


def reverse_backfill(apps, schema_editor):
    NonconformanceRecord = apps.get_model("agents_api", "NonconformanceRecord")
    Project = apps.get_model("core", "Project")

    # Find projects that are auto-created NCR projects
    auto_projects = Project.objects.filter(
        tags__contains=["auto-created"],
    ).filter(
        tags__contains=["ncr"],
    )

    count = NonconformanceRecord.objects.filter(
        project__in=auto_projects,
    ).update(project=None)

    if count:
        print(f"\n  Unlinked {count} NCR(s) from auto-created projects.")


class Migration(migrations.Migration):

    dependencies = [
        ("agents_api", "0039_auditfinding_evidence_auditfinding_status_and_more"),
        ("core", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(backfill_ncr_projects, reverse_backfill),
    ]

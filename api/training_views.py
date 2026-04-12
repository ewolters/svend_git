"""
Training center management — staff-only internal API.

Manages ILSSI partner training centers, programs, and student enrollments.
All endpoints require staff/internal access (IsInternalUser).

<!-- impl: api/training_views.py:list_centers -->
<!-- impl: api/training_views.py:create_center -->
<!-- impl: api/training_views.py:list_programs -->
<!-- impl: api/training_views.py:create_program -->
<!-- impl: api/training_views.py:batch_enroll -->
<!-- impl: api/training_views.py:batch_graduate -->
"""

import logging

from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from accounts.constants import Tier
from api.internal_views import IsInternalUser
from core.models import StudentEnrollment, TrainingCenter, TrainingProgram

User = get_user_model()
logger = logging.getLogger("svend.training")


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def _serialize_center(c):
    return {
        "id": str(c.id),
        "name": c.name,
        "country": c.country,
        "contact_name": c.contact_name,
        "contact_email": c.contact_email,
        "website": c.website,
        "instructor_email": c.instructor.email if c.instructor else None,
        "is_ilssi_partner": c.is_ilssi_partner,
        "is_ngo": c.is_ngo,
        "notes": c.notes,
        "total_students": c.total_students,
        "active_programs": c.active_programs.count(),
        "created_at": c.created_at.isoformat(),
    }


def _serialize_program(p):
    return {
        "id": str(p.id),
        "center_id": str(p.center_id),
        "center_name": p.center.name,
        "title": p.title,
        "description": p.description,
        "status": p.status,
        "start_date": str(p.start_date) if p.start_date else None,
        "end_date": str(p.end_date) if p.end_date else None,
        "region": p.region,
        "enrolled_count": p.enrolled_count,
        "graduated_count": p.graduated_count,
        "created_at": p.created_at.isoformat(),
    }


def _serialize_enrollment(e):
    return {
        "id": str(e.id),
        "user_id": str(e.user_id),
        "user_email": e.user.email,
        "program_id": str(e.program_id),
        "program_title": e.program.title,
        "status": e.status,
        "enrolled_at": e.enrolled_at.isoformat(),
        "graduated_at": e.graduated_at.isoformat() if e.graduated_at else None,
        "conversion_deadline": (e.conversion_deadline.isoformat() if e.conversion_deadline else None),
        "converted_at": e.converted_at.isoformat() if e.converted_at else None,
    }


# ---------------------------------------------------------------------------
# Training Centers
# ---------------------------------------------------------------------------


@api_view(["GET", "POST"])
@permission_classes([IsInternalUser])
def list_create_centers(request):
    """
    GET  — List all training centers.
    POST — Create a new training center.
    """
    if request.method == "GET":
        centers = TrainingCenter.objects.all()
        return Response({"centers": [_serialize_center(c) for c in centers]})

    name = request.data.get("name", "").strip()
    if not name:
        return Response({"error": "name is required"}, status=400)

    center = TrainingCenter.objects.create(
        name=name,
        country=request.data.get("country", "").upper()[:2],
        contact_name=request.data.get("contact_name", ""),
        contact_email=request.data.get("contact_email", ""),
        website=request.data.get("website", ""),
        is_ilssi_partner=request.data.get("is_ilssi_partner", True),
        is_ngo=request.data.get("is_ngo", False),
        notes=request.data.get("notes", ""),
    )

    # Link instructor if email provided
    instructor_email = request.data.get("instructor_email", "").strip()
    if instructor_email:
        try:
            instructor = User.objects.get(email=instructor_email)
            center.instructor = instructor
            center.save(update_fields=["instructor"])
            # Upgrade instructor to Enterprise
            if instructor.tier != Tier.ENTERPRISE:
                instructor.tier = Tier.ENTERPRISE
                instructor.save(update_fields=["tier"])
                logger.info("Upgraded instructor %s to Enterprise", instructor.email)
        except User.DoesNotExist:
            pass  # They'll register later, can be linked then

    logger.info("Training center created: %s (%s)", center.name, center.country)
    return Response(_serialize_center(center), status=201)


# ---------------------------------------------------------------------------
# Training Programs
# ---------------------------------------------------------------------------


@api_view(["GET", "POST"])
@permission_classes([IsInternalUser])
def list_create_programs(request):
    """
    GET  — List programs. Optional ?center_id= filter.
    POST — Create a new program under a center.
    """
    if request.method == "GET":
        programs = TrainingProgram.objects.select_related("center").all()
        center_id = request.query_params.get("center_id")
        if center_id:
            programs = programs.filter(center_id=center_id)
        return Response({"programs": [_serialize_program(p) for p in programs]})

    center_id = request.data.get("center_id")
    title = request.data.get("title", "").strip()
    if not center_id or not title:
        return Response({"error": "center_id and title are required"}, status=400)

    try:
        center = TrainingCenter.objects.get(id=center_id)
    except TrainingCenter.DoesNotExist:
        return Response({"error": "Training center not found"}, status=404)

    program = TrainingProgram.objects.create(
        center=center,
        title=title,
        description=request.data.get("description", ""),
        status=request.data.get("status", "planned"),
        start_date=request.data.get("start_date"),
        end_date=request.data.get("end_date"),
        region=request.data.get("region", "us"),
    )

    logger.info("Training program created: %s at %s", program.title, center.name)
    return Response(_serialize_program(program), status=201)


# ---------------------------------------------------------------------------
# Enrollments
# ---------------------------------------------------------------------------


@api_view(["GET"])
@permission_classes([IsInternalUser])
def list_enrollments(request):
    """List enrollments. Optional ?program_id= or ?center_id= filter."""
    enrollments = StudentEnrollment.objects.select_related("user", "program", "program__center").all()

    program_id = request.query_params.get("program_id")
    if program_id:
        enrollments = enrollments.filter(program_id=program_id)

    center_id = request.query_params.get("center_id")
    if center_id:
        enrollments = enrollments.filter(program__center_id=center_id)

    status = request.query_params.get("status")
    if status:
        enrollments = enrollments.filter(status=status)

    return Response({"enrollments": [_serialize_enrollment(e) for e in enrollments]})


@api_view(["POST"])
@permission_classes([IsInternalUser])
def batch_enroll(request):
    """
    Batch-create student accounts and enroll them in a program.

    POST body: {
        "program_id": uuid,
        "students": [
            {"email": "...", "name": "..."},
            ...
        ]
    }

    For each student:
    - If account exists: enroll and upgrade to Pro
    - If account doesn't exist: create account, set to Pro, enroll
    """
    program_id = request.data.get("program_id")
    students = request.data.get("students", [])

    if not program_id:
        return Response({"error": "program_id is required"}, status=400)
    if not students:
        return Response({"error": "students list is required"}, status=400)

    try:
        program = TrainingProgram.objects.get(id=program_id)
    except TrainingProgram.DoesNotExist:
        return Response({"error": "Program not found"}, status=404)

    results = []
    for student_data in students:
        email = student_data.get("email", "").strip().lower()
        name = student_data.get("name", "").strip()

        if not email:
            results.append({"email": email, "status": "skipped", "reason": "no email"})
            continue

        # Find or create user
        user, created = User.objects.get_or_create(
            email=email,
            defaults={
                "username": email.split("@")[0],
                "display_name": name,
                "role": "student",
                "industry": "education",
                "is_email_verified": True,
            },
        )

        if created:
            # Set a random password — student will need to reset
            import secrets

            user.set_password(secrets.token_urlsafe(16))
            user.save()

        # Upgrade to Pro (free during enrollment)
        if user.tier in (Tier.FREE, "free"):
            user.tier = Tier.PRO
            user.save(update_fields=["tier"])

        # Create enrollment (idempotent)
        enrollment, enrolled_new = StudentEnrollment.objects.get_or_create(
            user=user,
            program=program,
            defaults={
                "status": StudentEnrollment.Status.ENROLLED,
                "created_by": request.user,
            },
        )

        results.append(
            {
                "email": email,
                "name": name,
                "status": "enrolled" if enrolled_new else "already_enrolled",
                "account_created": created,
                "enrollment_id": str(enrollment.id),
            }
        )

    enrolled_count = sum(1 for r in results if r["status"] == "enrolled")
    logger.info(
        "Batch enrollment: %d/%d students enrolled in %s",
        enrolled_count,
        len(students),
        program.title,
    )

    return Response(
        {
            "program": _serialize_program(program),
            "results": results,
            "enrolled": enrolled_count,
            "already_enrolled": sum(1 for r in results if r["status"] == "already_enrolled"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
        }
    )


@api_view(["POST"])
@permission_classes([IsInternalUser])
def batch_graduate(request):
    """
    Batch-graduate students in a program.

    POST body: {
        "program_id": uuid,
        "student_emails": ["email1", "email2", ...]  (optional — if empty, graduates all enrolled)
    }

    Sets 30-day conversion window for alumni discount.
    """
    program_id = request.data.get("program_id")
    if not program_id:
        return Response({"error": "program_id is required"}, status=400)

    try:
        program = TrainingProgram.objects.get(id=program_id)
    except TrainingProgram.DoesNotExist:
        return Response({"error": "Program not found"}, status=404)

    student_emails = request.data.get("student_emails", [])

    enrollments = StudentEnrollment.objects.filter(
        program=program,
        status=StudentEnrollment.Status.ENROLLED,
    ).select_related("user")

    if student_emails:
        enrollments = enrollments.filter(user__email__in=[e.lower() for e in student_emails])

    graduated = 0
    results = []
    for enrollment in enrollments:
        enrollment.graduate()
        graduated += 1
        results.append(
            {
                "email": enrollment.user.email,
                "status": "graduated",
                "conversion_deadline": enrollment.conversion_deadline.isoformat(),
            }
        )

    # Mark program as completed if all students graduated
    remaining = StudentEnrollment.objects.filter(
        program=program,
        status=StudentEnrollment.Status.ENROLLED,
    ).count()
    if remaining == 0 and graduated > 0:
        program.status = TrainingProgram.Status.COMPLETED
        program.save(update_fields=["status"])

    logger.info("Batch graduation: %d students graduated from %s", graduated, program.title)

    return Response(
        {
            "program": _serialize_program(program),
            "graduated": graduated,
            "results": results,
        }
    )

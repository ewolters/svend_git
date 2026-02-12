"""Publish blog posts that have reached their scheduled time."""

from django.core.management.base import BaseCommand
from django.utils import timezone

from api.models import BlogPost


class Command(BaseCommand):
    help = "Publish blog posts whose scheduled_at time has passed."

    def handle(self, *args, **options):
        now = timezone.now()
        due = BlogPost.objects.filter(status="scheduled", scheduled_at__lte=now)
        count = 0
        for post in due:
            post.status = BlogPost.Status.PUBLISHED
            post.published_at = post.scheduled_at
            post.save()
            count += 1
            self.stdout.write(f"  Published: {post.title}")

        if count:
            self.stdout.write(self.style.SUCCESS(f"Published {count} scheduled post(s)."))
        else:
            self.stdout.write("No posts due for publishing.")

"""Tempora tasks for the API app."""

import logging

from django.utils import timezone

from tempora.core import task
from tempora.types import QueueType, TaskPriority

logger = logging.getLogger(__name__)


@task(
    "api.publish_scheduled_posts",
    queue=QueueType.CORE,
    priority=TaskPriority.LOW,
    timeout_seconds=30,
    max_attempts=1,
)
def publish_scheduled_posts(payload, context):
    """Publish blog posts whose scheduled_at time has passed."""
    from api.models import BlogPost

    now = timezone.now()
    due = BlogPost.objects.filter(status="scheduled", scheduled_at__lte=now)
    count = 0
    for post in due:
        post.status = BlogPost.Status.PUBLISHED
        post.published_at = post.scheduled_at
        post.save()
        count += 1
        logger.info(f"Published scheduled post: {post.title}")

    return {"published": count}


# ---------------------------------------------------------------------------
# Onboarding email drip
# ---------------------------------------------------------------------------

LEARNING_PATH_LABELS = {
    "quality_engineer": "Quality Engineering",
    "manager": "Management & Reporting",
    "researcher": "Research & Advanced Statistics",
    "analyst": "Data Analysis",
    "student": "Learning Statistics",
    "beginner": "Getting Started",
}

# Email content keyed by email_key. Each returns (subject, body_html) given user + survey.
def _email_welcome(user, survey):
    name = user.display_name or user.username
    path_label = LEARNING_PATH_LABELS.get(survey.learning_path, "Getting Started") if survey else "Getting Started"
    return (
        "Welcome to Svend!",
        f"""<h2>Hey {name},</h2>
<p>Thanks for signing up. We built Svend because statistical tools shouldn't cost thousands of dollars or require a PhD to use.</p>

<p>Based on your profile, we've set you up on the <strong>{path_label}</strong> track. Everything you see is tailored to help you get results fast.</p>

<p><strong>Your first 3 things to try:</strong></p>
<ol>
<li><a href="https://svend.ai/app/dsw/" style="color:#4a9f6e;">Run a quick analysis</a> in the Decision Science Workbench</li>
<li>Check out the <a href="https://svend.ai/app/learn/" style="color:#4a9f6e;">Learning Center</a> for guided tutorials</li>
<li>Try an <a href="https://svend.ai/app/spc/" style="color:#4a9f6e;">SPC control chart</a> with your own data</li>
</ol>

<p>You get 5 free runs per day. If you hit that limit quickly, that's a good sign you should check out our <a href="https://svend.ai/#pricing" style="color:#4a9f6e;">paid plans</a> starting at $19/month.</p>

<p>Reply to this email anytime. I read everything.</p>
<p>-- Eric, Founder</p>""",
    )


def _email_getting_started(user, survey):
    name = user.display_name or user.username
    goal = survey.primary_goal if survey else ""

    # Personalize based on primary goal
    tip = ""
    if goal == "spc":
        tip = """<p><strong>Since you're interested in SPC:</strong> Head to the <a href="https://svend.ai/app/spc/" style="color:#4a9f6e;">SPC module</a> and paste in some process data. Svend will generate X-bar/R, I-MR, or p-charts automatically and flag any out-of-control points. No setup needed.</p>"""
    elif goal == "doe":
        tip = """<p><strong>Since you're interested in DOE:</strong> The <a href="https://svend.ai/app/experimenter/" style="color:#4a9f6e;">Experimenter</a> can design full factorial, fractional factorial, and response surface experiments. Describe your factors and it handles the rest.</p>"""
    elif goal == "analysis":
        tip = """<p><strong>Since you want to analyze data:</strong> The <a href="https://svend.ai/app/dsw/" style="color:#4a9f6e;">DSW</a> supports 60+ statistical tests. Paste your data, describe what you're looking for, and the AI picks the right test.</p>"""
    elif goal == "replace_tool":
        tip = """<p><strong>Switching from another tool?</strong> Svend handles most of what Minitab and JMP do at a fraction of the cost. The <a href="https://svend.ai/app/dsw/" style="color:#4a9f6e;">DSW</a> covers hypothesis tests, regression, ANOVA, capability studies, and more.</p>"""
    elif goal == "reporting":
        tip = """<p><strong>For reports:</strong> Every analysis in Svend can be exported as a PDF. Run your analysis, then hit the export button. Clean formatting, ready for management review.</p>"""
    else:
        tip = """<p>Start with the <a href="https://svend.ai/app/dsw/" style="color:#4a9f6e;">Decision Science Workbench</a> and describe what you're working on. The AI agent will pick the right tools for you.</p>"""

    return (
        f"Getting started, {name}",
        f"""<h2>A quick tour of what matters</h2>
{tip}
<p><strong>Pro tip:</strong> You can describe your problem in plain English. Try something like "I have measurement data from two machines and need to know if they're different." The agent will figure out the right statistical test.</p>

<p><a href="https://svend.ai/app/" style="color:#4a9f6e; font-weight:600;">Open Svend &rarr;</a></p>""",
    )


def _email_tips(user, survey):
    name = user.display_name or user.username
    confidence = survey.confidence_stats if survey else 3

    if confidence <= 2:
        # Low confidence: focus on learning
        content = """<h2>Statistics doesn't have to be intimidating</h2>
<p>A lot of people feel unsure about statistics. That's normal and honestly, most tools make it worse by throwing jargon at you.</p>
<p>Svend is different. You describe your problem, and the AI handles the method selection. But if you want to understand what's happening under the hood:</p>
<ul>
<li>The <a href="https://svend.ai/app/learn/" style="color:#4a9f6e;">Learning Center</a> has bite-sized modules on the most common tests</li>
<li>Every analysis result includes a plain-English interpretation</li>
<li>Our <a href="https://svend.ai/blog/" style="color:#4a9f6e;">blog</a> covers practical topics like Cpk vs Ppk, control charts, and choosing the right test</li>
</ul>"""
    elif confidence >= 4:
        # High confidence: power user tips
        content = """<h2>Power user tips</h2>
<p>Since you're comfortable with statistics, here are some features you might not have found yet:</p>
<ul>
<li><strong>Experimenter:</strong> Full DOE support including response surface methodology and optimal designs</li>
<li><strong>Calculators:</strong> Sample size, tolerance intervals, reliability, and more</li>
<li><strong>A3 Reports:</strong> Structure your problem-solving with Toyota-style A3 thinking</li>
<li><strong>Projects:</strong> Link analyses to hypotheses and build an evidence chain with Bayesian updating</li>
</ul>"""
    else:
        content = """<h2>3 features most people miss</h2>
<ul>
<li><strong>Calculators:</strong> Quick sample size, Cpk, confidence interval calculators at <a href="https://svend.ai/app/calculators/" style="color:#4a9f6e;">svend.ai/app/calculators</a></li>
<li><strong>Learning Center:</strong> Short modules on the statistics you actually use at work</li>
<li><strong>Export to PDF:</strong> Every analysis can be exported as a clean report</li>
</ul>"""

    return (
        f"3 things you should try, {name}",
        content,
    )


def _email_learning_path(user, survey):
    name = user.display_name or user.username
    path = survey.learning_path if survey else "beginner"

    paths = {
        "quality_engineer": {
            "title": "Your Quality Engineering Path",
            "items": [
                ("SPC Control Charts", "/app/spc/", "Monitor process stability with X-bar/R, I-MR, and attribute charts"),
                ("Capability Analysis", "/app/dsw/", "Calculate Cpk, Ppk, and process performance indices"),
                ("Gage R&R", "/app/dsw/", "Validate your measurement systems"),
                ("DOE", "/app/experimenter/", "Optimize processes with designed experiments"),
            ],
        },
        "manager": {
            "title": "Your Management Dashboard Path",
            "items": [
                ("Quick Analysis", "/app/dsw/", "Get answers from data without needing to pick the right test"),
                ("SPC Dashboards", "/app/spc/", "Monitor key process metrics at a glance"),
                ("A3 Reports", "/app/a3/", "Structure problem-solving for your team"),
                ("PDF Export", "/app/dsw/", "Generate clean reports for stakeholders"),
            ],
        },
        "researcher": {
            "title": "Your Research Path",
            "items": [
                ("Advanced Analysis", "/app/dsw/", "60+ statistical tests including nonparametric methods"),
                ("DOE", "/app/experimenter/", "Full factorial, fractional factorial, RSM, and optimal designs"),
                ("Hypothesis Engine", "/app/projects/", "Link evidence to hypotheses with Bayesian updating"),
                ("Forecast", "/app/forecast/", "Time series analysis and forecasting"),
            ],
        },
        "analyst": {
            "title": "Your Data Analysis Path",
            "items": [
                ("Statistical Testing", "/app/dsw/", "Run the right test automatically based on your data"),
                ("Visualization", "/app/dsw/", "Interactive charts and distribution plots"),
                ("Process Monitoring", "/app/spc/", "Set up ongoing control charts"),
                ("Calculators", "/app/calculators/", "Quick calculations for sample size, confidence intervals, and more"),
            ],
        },
        "student": {
            "title": "Your Learning Path",
            "items": [
                ("Learning Center", "/app/learn/", "Guided modules on core statistical concepts"),
                ("Practice Analysis", "/app/dsw/", "Try analyses with example datasets"),
                ("Blog", "/blog/", "Practical articles on statistics in industry"),
                ("Calculators", "/app/calculators/", "Build intuition with interactive tools"),
            ],
        },
    }

    path_data = paths.get(path, paths.get("student"))

    items_html = ""
    for title, url, desc in path_data["items"]:
        items_html += f'<li><a href="https://svend.ai{url}" style="color:#4a9f6e;font-weight:600;">{title}</a> — {desc}</li>\n'

    return (
        f"{path_data['title']}",
        f"""<h2>Hey {name},</h2>
<p>Based on your background, here's a recommended path through Svend:</p>
<ol>
{items_html}
</ol>
<p>Work through these in order and you'll have a solid handle on what Svend can do for you. Each one takes about 5 minutes.</p>
<p><a href="https://svend.ai/app/" style="color:#4a9f6e;font-weight:600;">Get started &rarr;</a></p>""",
    )


def _email_checkin(user, survey):
    name = user.display_name or user.username
    queries = user.total_queries

    if queries == 0:
        content = f"""<h2>Hey {name},</h2>
<p>I noticed you haven't run any analyses yet. No pressure, but I wanted to make sure everything's working for you.</p>
<p>If you're not sure where to start, just go to the <a href="https://svend.ai/app/dsw/" style="color:#4a9f6e;">workbench</a> and describe what you're trying to figure out. Something like:</p>
<ul>
<li>"I have yield data from two production lines and want to know if they're different"</li>
<li>"Check if my process is in control"</li>
<li>"What sample size do I need to detect a 10% improvement?"</li>
</ul>
<p>Or just reply to this email and tell me what you're working on. Happy to point you in the right direction.</p>"""
    elif queries < 5:
        content = f"""<h2>Hey {name},</h2>
<p>You've run {queries} analyses so far. How's it going?</p>
<p>If something isn't working the way you expected, or if there's a feature you wish existed, just reply to this email. I'm building this based on what users actually need.</p>
<p>If you're finding Svend useful and hitting the daily limit, the <a href="https://svend.ai/#pricing" style="color:#4a9f6e;">Founder plan at $19/month</a> gives you 50 runs/day and locks in that price forever.</p>"""
    else:
        content = f"""<h2>Hey {name},</h2>
<p>You've run {queries} analyses this week. Looks like you're getting good use out of Svend.</p>
<p>Quick question: is there anything that would make it even more useful for your work? A specific test, a feature, a workflow? Reply and let me know.</p>
<p>If you haven't already, the <a href="https://svend.ai/#pricing" style="color:#4a9f6e;">Founder plan</a> is still available at $19/month (locked forever). Only {'{remaining}'} of 100 spots left.</p>"""

    return (
        f"Quick check-in, {name}",
        content,
    )


EMAIL_BUILDERS = {
    "welcome": _email_welcome,
    "getting_started": _email_getting_started,
    "tips": _email_tips,
    "learning_path": _email_learning_path,
    "checkin": _email_checkin,
}


@task(
    "api.send_onboarding_email",
    queue=QueueType.CORE,
    priority=TaskPriority.NORMAL,
    timeout_seconds=30,
    max_attempts=3,
)
def send_onboarding_email(payload, context):
    """Send a single onboarding drip email."""
    from django.core.mail import send_mail
    from django.conf import settings as django_settings
    from accounts.models import User
    from api.models import OnboardingEmail, OnboardingSurvey
    from api.internal_views import EMAIL_TEMPLATE
    from accounts.constants import get_founder_availability

    user_id = payload.get("user_id")
    email_key = payload.get("email_key")

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        logger.warning(f"Onboarding email: user {user_id} not found")
        return {"error": "user_not_found"}

    if not user.email:
        return {"error": "no_email"}

    # Get survey data
    survey = None
    try:
        survey = OnboardingSurvey.objects.get(user=user)
    except OnboardingSurvey.DoesNotExist:
        pass

    # Build email content
    builder = EMAIL_BUILDERS.get(email_key)
    if not builder:
        logger.warning(f"Unknown email key: {email_key}")
        return {"error": f"unknown_key_{email_key}"}

    subject, body_html = builder(user, survey)

    # For checkin email, fill in founder availability
    if email_key == "checkin" and "{remaining}" in body_html:
        avail = get_founder_availability()
        body_html = body_html.replace("{remaining}", str(avail["remaining"]))

    full_html = EMAIL_TEMPLATE.format(body=body_html)

    try:
        send_mail(
            subject=subject,
            message="",
            from_email=django_settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            html_message=full_html,
        )
        # Mark as sent
        OnboardingEmail.objects.filter(
            user=user, email_key=email_key
        ).update(status="sent", sent_at=timezone.now())
        logger.info(f"Sent onboarding email '{email_key}' to {user.email}")
        return {"sent": True, "email_key": email_key}
    except Exception as e:
        OnboardingEmail.objects.filter(
            user=user, email_key=email_key
        ).update(status="failed")
        logger.error(f"Failed to send onboarding email '{email_key}' to {user.email}: {e}")
        return {"error": str(e)}


@task(
    "api.process_onboarding_drip",
    queue=QueueType.CORE,
    priority=TaskPriority.LOW,
    timeout_seconds=60,
    max_attempts=1,
)
def process_onboarding_drip(payload, context):
    """Process pending onboarding drip emails that are due."""
    from api.models import OnboardingEmail

    now = timezone.now()
    due = OnboardingEmail.objects.filter(
        status="pending",
        scheduled_for__lte=now,
    ).select_related("user")

    count = 0
    for email in due[:50]:  # batch cap
        try:
            from tempora.scheduler import schedule_task
            schedule_task(
                name=f"onboarding_{email.email_key}_{email.user_id}",
                func="api.send_onboarding_email",
                args={
                    "user_id": str(email.user_id),
                    "email_key": email.email_key,
                },
                delay_seconds=0,
                priority=2,
                queue="core",
            )
            count += 1
        except Exception as e:
            logger.error(f"Failed to schedule drip email: {e}")

    return {"scheduled": count}

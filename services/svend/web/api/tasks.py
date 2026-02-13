"""Tempora tasks for the API app."""

import logging
from datetime import timedelta

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

    if getattr(user, "email_opted_out", False):
        OnboardingEmail.objects.filter(
            user=user, email_key=email_key
        ).update(status="skipped")
        return {"skipped": "opted_out"}

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

    from api.views import make_unsubscribe_url
    unsub_url = make_unsubscribe_url(user)
    full_html = EMAIL_TEMPLATE.format(body=body_html, unsub_url=unsub_url)

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


# ---------------------------------------------------------------------------
# Lifecycle email templates (for automation rules)
# ---------------------------------------------------------------------------

def _lifecycle_activation(user):
    """Quick start guide for users who signed up but haven't queried."""
    name = user.display_name or user.username
    return (
        f"Getting the most out of Svend, {name}",
        f"""<h2>Hey {name},</h2>
<p>You signed up a few days ago but haven't run an analysis yet. No worries — here's how to get started in under a minute:</p>
<ol>
<li>Go to the <a href="https://svend.ai/app/dsw/" style="color:#4a9f6e;">Decision Science Workbench</a></li>
<li>Describe your problem in plain English, or paste some data</li>
<li>Svend picks the right statistical test and runs it for you</li>
</ol>
<p>Try something like: <em>"Compare the means of these two groups: [34, 38, 42, 37] vs [45, 41, 49, 43]"</em></p>
<p>Reply anytime if you have questions.</p>
<p>-- Eric, Founder</p>""",
    )


def _lifecycle_inactive_nudge(user):
    """Nudge for users inactive 7+ days."""
    name = user.display_name or user.username
    return (
        f"We've added new features, {name}",
        f"""<h2>Hey {name},</h2>
<p>It's been a little while since you've been on Svend. We've been shipping improvements:</p>
<ul>
<li><strong>60+ statistical analyses</strong> in the Decision Science Workbench</li>
<li><strong>SPC control charts</strong> with automatic out-of-control detection</li>
<li><strong>DOE</strong> for designing experiments with optimal factor combinations</li>
<li><strong>Learning Center</strong> with guided tutorials</li>
</ul>
<p><a href="https://svend.ai/app/" style="color:#4a9f6e;font-weight:600;">Open Svend &rarr;</a></p>
<p>If something wasn't working for you, I'd love to hear about it. Just reply.</p>
<p>-- Eric</p>""",
    )


def _lifecycle_upgrade_nudge(user):
    """For users approaching their daily query limit."""
    name = user.display_name or user.username
    return (
        f"You're a power user, {name}",
        f"""<h2>Hey {name},</h2>
<p>You're hitting your daily analysis limit regularly — that means you're getting real value from Svend.</p>
<p>The <a href="https://svend.ai/#pricing" style="color:#4a9f6e;">Founder plan</a> gives you <strong>10x the daily limit</strong> for $19/month — and that price is locked in forever, even as we raise prices later.</p>
<p><strong>What you get:</strong></p>
<ul>
<li>50 analyses/day (vs 5 on free)</li>
<li>ML model training and advanced DOE</li>
<li>Priority processing</li>
<li>Early access to new features</li>
</ul>
<p><a href="https://svend.ai/#pricing" style="color:#4a9f6e;font-weight:600;">See plans &rarr;</a></p>
<p>-- Eric</p>""",
    )


def _lifecycle_churn_prevention(user):
    """For users whose subscription is set to cancel."""
    name = user.display_name or user.username
    return (
        f"Before you go, {name}",
        f"""<h2>Hey {name},</h2>
<p>I saw that you've scheduled your subscription to cancel. I get it — not every tool is right for everyone.</p>
<p>But before you go, I wanted to ask: <strong>is there something specific that didn't work for you?</strong></p>
<p>I'm a solo founder building this based on real user needs. If there's a feature missing or something that was confusing, I want to fix it. Just reply to this email.</p>
<p>If you change your mind, your account and all your data will still be here. You can reactivate anytime from your <a href="https://svend.ai/app/settings/" style="color:#4a9f6e;">account settings</a>.</p>
<p>-- Eric</p>""",
    )


def _lifecycle_feature_discovery(user, feature="doe"):
    """Nudge paid users toward features they haven't tried."""
    name = user.display_name or user.username
    features = {
        "doe": {
            "title": "Design of Experiments",
            "desc": "Optimize processes by testing multiple factors simultaneously — more efficient than one-at-a-time experiments.",
            "url": "https://svend.ai/app/experimenter/",
            "cta": "Try DOE",
        },
        "spc": {
            "title": "Statistical Process Control",
            "desc": "Monitor process stability with control charts. Detects shifts before they become defects.",
            "url": "https://svend.ai/app/spc/",
            "cta": "Try SPC",
        },
        "forecast": {
            "title": "Time Series Forecasting",
            "desc": "Predict trends using your historical data. Supports multiple methods automatically.",
            "url": "https://svend.ai/app/forecast/",
            "cta": "Try Forecasting",
        },
    }
    f = features.get(feature, features["doe"])
    return (
        f"Have you tried {f['title']}?",
        f"""<h2>Hey {name},</h2>
<p>You're using Svend for analysis — great. But there's a feature you haven't tried yet that might be useful:</p>
<h3>{f['title']}</h3>
<p>{f['desc']}</p>
<p><a href="{f['url']}" style="color:#4a9f6e;font-weight:600;">{f['cta']} &rarr;</a></p>
<p>-- Eric</p>""",
    )


def _lifecycle_milestone(user, count=100):
    """Celebrate query milestones."""
    name = user.display_name or user.username
    return (
        f"Milestone: {count} analyses!",
        f"""<h2>Hey {name},</h2>
<p>You've just hit <strong>{count} analyses</strong> on Svend. That's impressive.</p>
<p>You're clearly getting real work done with this tool, and that's exactly what it's built for.</p>
<p>If you're on the free plan, the <a href="https://svend.ai/#pricing" style="color:#4a9f6e;">Founder plan at $19/month</a> would give you 10x the daily limit. If you're already a paying customer — thank you. You're literally making this possible.</p>
<p>Keep going. And if you ever want a feature added, just reply.</p>
<p>-- Eric</p>""",
    )


def _lifecycle_winback(user):
    """Win-back for formerly paid users inactive 45+ days."""
    name = user.display_name or user.username
    return (
        f"We've been improving, {name}",
        f"""<h2>Hey {name},</h2>
<p>It's been a while since you've used Svend. A lot has changed since you were last here:</p>
<ul>
<li>60+ statistical analyses with AI-guided test selection</li>
<li>SPC with automatic control chart selection</li>
<li>Full DOE including response surface methodology</li>
<li>Learning center with guided tutorials</li>
<li>Faster, more accurate results across the board</li>
</ul>
<p>Your account is still active. Everything's where you left it.</p>
<p><a href="https://svend.ai/app/" style="color:#4a9f6e;font-weight:600;">Come back and take a look &rarr;</a></p>
<p>-- Eric</p>""",
    )


LIFECYCLE_BUILDERS = {
    "activation": _lifecycle_activation,
    "inactive_nudge": _lifecycle_inactive_nudge,
    "upgrade_nudge": _lifecycle_upgrade_nudge,
    "churn_prevention": _lifecycle_churn_prevention,
    "feature_discovery": _lifecycle_feature_discovery,
    "milestone": _lifecycle_milestone,
    "winback": _lifecycle_winback,
}


# ---------------------------------------------------------------------------
# Automation tasks
# ---------------------------------------------------------------------------

def _send_lifecycle_email(user, template_key, **kwargs):
    """Send a lifecycle email using the standard email infrastructure."""
    from django.conf import settings as django_settings
    from django.core.mail import send_mail

    from api.internal_views import EMAIL_TEMPLATE
    from api.models import EmailCampaign, EmailRecipient

    builder = LIFECYCLE_BUILDERS.get(template_key)
    if not builder:
        logger.warning("Unknown lifecycle template: %s", template_key)
        return False

    if not user.email:
        return False

    if getattr(user, "email_opted_out", False):
        return False

    try:
        subject, body_html = builder(user, **kwargs)
    except TypeError:
        subject, body_html = builder(user)

    # Create campaign for traceability
    campaign = EmailCampaign.objects.create(
        subject=subject,
        body_md=body_html,
        target=f"automation:{template_key}",
        recipient_count=1,
    )
    rcpt = EmailRecipient.objects.create(
        campaign=campaign,
        user=user,
        email=user.email,
    )

    # Link rewriting for click tracking
    import re
    def _track_link(match):
        url = match.group(1)
        return f'href="https://svend.ai/api/email/click/{rcpt.id}/?url={url}"'
    tracked = re.sub(r'href="(https?://[^"]+)"', _track_link, body_html)

    # Tracking pixel + unsubscribe
    from api.views import make_unsubscribe_url
    pixel = f'<img src="https://svend.ai/api/email/open/{rcpt.id}/" width="1" height="1" style="display:none;" alt="">'
    unsub_url = make_unsubscribe_url(user)
    full_html = EMAIL_TEMPLATE.format(body=tracked + pixel, unsub_url=unsub_url)

    try:
        send_mail(
            subject=subject,
            message="",
            from_email=django_settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            html_message=full_html,
        )
        return True
    except Exception as e:
        rcpt.failed = True
        rcpt.save(update_fields=["failed"])
        logger.error("Lifecycle email failed for %s: %s", user.email, e)
        return False


@task(
    "api.process_automations",
    queue=QueueType.CORE,
    priority=TaskPriority.NORMAL,
    timeout_seconds=120,
    max_attempts=1,
)
def process_automations(payload, context):
    """Evaluate all active automation rules and fire matching actions."""
    from accounts.constants import TIER_LIMITS
    from accounts.models import Subscription, User
    from api.models import AutomationLog, AutomationRule

    now = timezone.now()
    rules = AutomationRule.objects.filter(is_active=True)
    staff_ids = set(User.objects.filter(is_staff=True).values_list("id", flat=True))
    fired = 0

    for rule in rules:
        cfg = rule.trigger_config or {}
        matched_users = []

        if rule.trigger == "signup_no_query":
            days = cfg.get("days", 3)
            cutoff = now - timedelta(days=days)
            matched_users = list(
                User.objects.filter(
                    date_joined__lte=cutoff,
                    total_queries=0,
                    is_active=True,
                    email_opted_out=False,
                ).exclude(id__in=staff_ids)
            )

        elif rule.trigger == "inactive_days":
            days = cfg.get("days", 7)
            cutoff = now - timedelta(days=days)
            matched_users = list(
                User.objects.filter(
                    last_active_at__lte=cutoff,
                    total_queries__gt=0,
                    is_active=True,
                    email_opted_out=False,
                ).exclude(id__in=staff_ids)
            )

        elif rule.trigger == "query_limit_near":
            threshold_pct = cfg.get("threshold", 80)
            for user in User.objects.filter(is_active=True, tier="free", email_opted_out=False).exclude(id__in=staff_ids):
                limit = TIER_LIMITS.get(user.tier, 5)
                if limit > 0 and user.queries_today >= (limit * threshold_pct / 100):
                    matched_users.append(user)

        elif rule.trigger == "churn_signal":
            sub_user_ids = Subscription.objects.filter(
                cancel_at_period_end=True,
                status="active",
            ).values_list("user_id", flat=True)
            matched_users = list(
                User.objects.filter(
                    id__in=sub_user_ids,
                    is_active=True,
                    email_opted_out=False,
                ).exclude(id__in=staff_ids)
            )

        elif rule.trigger == "milestone":
            threshold = cfg.get("count", 100)
            matched_users = list(
                User.objects.filter(
                    total_queries__gte=threshold,
                    is_active=True,
                    email_opted_out=False,
                ).exclude(id__in=staff_ids)
            )

        elif rule.trigger == "feature_unused":
            # Check paid users who haven't used a specific domain
            feature = cfg.get("feature", "doe")
            days = cfg.get("days", 14)
            cutoff = now - timedelta(days=days)
            from chat.models import UsageLog
            paid_users = User.objects.filter(
                tier__in=["founder", "pro", "team", "enterprise"],
                is_active=True,
                email_opted_out=False,
            ).exclude(id__in=staff_ids)
            for user in paid_users:
                recent = UsageLog.objects.filter(
                    user=user,
                    date__gte=cutoff.date(),
                ).order_by("-date").first()
                if recent and recent.domain_counts:
                    if feature not in recent.domain_counts:
                        matched_users.append(user)
                elif recent:
                    matched_users.append(user)

        # Filter by cooldown
        for user in matched_users:
            # Check cooldown
            if rule.cooldown_hours > 0:
                cooldown_since = now - timedelta(hours=rule.cooldown_hours)
                already_fired = AutomationLog.objects.filter(
                    rule=rule,
                    user=user,
                    fired_at__gte=cooldown_since,
                ).exists()
                if already_fired:
                    continue

            # For milestone: check if ever fired (never re-fire)
            if rule.trigger == "milestone":
                ever_fired = AutomationLog.objects.filter(rule=rule, user=user).exists()
                if ever_fired:
                    continue

            # Execute action
            result = "skipped"
            action_taken = ""

            if rule.action == "send_email":
                template = (rule.action_config or {}).get("template", rule.trigger)
                extra = {}
                if rule.trigger == "feature_unused":
                    extra["feature"] = cfg.get("feature", "doe")
                if rule.trigger == "milestone":
                    extra["count"] = cfg.get("count", 100)

                success = _send_lifecycle_email(user, template, **extra)
                action_taken = f"email:{template} to {user.email}"
                result = "success" if success else "failed"

            elif rule.action == "internal_alert":
                msg = (rule.action_config or {}).get("message", "Alert for {username}")
                action_taken = msg.replace("{username}", user.username)
                result = "success"

            AutomationLog.objects.create(
                rule=rule,
                user=user,
                action_taken=action_taken,
                result=result,
            )
            if result == "success":
                fired += 1

        # Update rule stats
        if matched_users:
            rule.times_fired += fired
            rule.last_fired_at = now
            rule.save(update_fields=["times_fired", "last_fired_at"])

    return {"rules_checked": rules.count(), "actions_fired": fired}


@task(
    "api.evaluate_experiments",
    queue=QueueType.CORE,
    priority=TaskPriority.LOW,
    timeout_seconds=60,
    max_attempts=1,
)
def evaluate_experiments(payload, context):
    """Evaluate all running experiments for significance."""
    from api.experiments import evaluate_experiment
    from api.models import Experiment

    experiments = Experiment.objects.filter(status="running")
    evaluated = 0
    concluded = 0

    for exp in experiments:
        results = evaluate_experiment(exp)
        evaluated += 1
        if exp.status == "concluded":
            concluded += 1
            logger.info("Experiment '%s' concluded. Winner: %s", exp.name, exp.winner)

    return {"evaluated": evaluated, "concluded": concluded}


@task(
    "api.claude_growth_review",
    queue=QueueType.CORE,
    priority=TaskPriority.LOW,
    timeout_seconds=300,
    max_attempts=1,
)
def claude_growth_review(payload, context):
    """Weekly Claude-powered growth review. Generates insights and recommendations."""
    import json

    import anthropic

    from api.internal_views import _build_data_snapshot
    from api.models import (
        AutomationLog,
        AutomationRule,
        AutopilotReport,
        EmailCampaign,
        Experiment,
    )

    # Gather data
    snapshot = _build_data_snapshot(days=7)

    # Experiment results
    exp_data = []
    for exp in Experiment.objects.filter(status__in=["running", "concluded"]).order_by("-created_at")[:10]:
        exp_data.append({
            "name": exp.name,
            "status": exp.status,
            "type": exp.experiment_type,
            "winner": exp.winner,
            "results": exp.results,
        })

    # Automation fire rates
    now = timezone.now()
    week_ago = now - timedelta(days=7)
    rule_stats = []
    for rule in AutomationRule.objects.filter(is_active=True):
        fires = AutomationLog.objects.filter(rule=rule, fired_at__gte=week_ago).count()
        successes = AutomationLog.objects.filter(
            rule=rule, fired_at__gte=week_ago, result="success"
        ).count()
        rule_stats.append({
            "name": rule.name,
            "trigger": rule.trigger,
            "fires_this_week": fires,
            "successes": successes,
        })

    # Email campaign stats
    campaign_stats = []
    for campaign in EmailCampaign.objects.filter(created_at__gte=week_ago).order_by("-created_at")[:10]:
        total = campaign.recipients.count()
        opened = campaign.recipients.filter(opened_at__isnull=False).count()
        clicked = campaign.recipients.filter(clicked_at__isnull=False).count()
        campaign_stats.append({
            "subject": campaign.subject,
            "target": campaign.target,
            "sent": total,
            "opened": opened,
            "clicked": clicked,
        })

    # Recent feedback
    from api.models import Feedback
    feedback_data = []
    for fb in Feedback.objects.filter(created_at__gte=week_ago).order_by("-created_at")[:20]:
        feedback_data.append({
            "category": fb.category,
            "message": fb.message,
            "page": fb.page,
            "user_tier": fb.user.tier if fb.user else "unknown",
            "status": fb.status,
        })

    prompt_data = {
        "weekly_snapshot": snapshot,
        "experiments": exp_data,
        "automation_rules": rule_stats,
        "email_campaigns": campaign_stats,
        "user_feedback": feedback_data,
    }

    system_prompt = """You are a growth advisor for Svend, a SaaS decision science platform for engineers and analysts.
Pricing: Free (5/day) / Founder $19/mo (50/day) / Pro $29/mo / Team $79/mo / Enterprise $199/mo.
Competing against Minitab ($1,851/yr) and JMP ($1,320-$8,400/yr).
Solo founder operation. Every recommendation must be actionable with zero employees.

Respond with ONLY valid JSON (no markdown, no code fences) in this structure:
{
  "insights": ["insight 1", "insight 2", ...],
  "recommendations": [
    {
      "type": "email|experiment|blog|rule_tweak|manual",
      "priority": "high|medium|low",
      "title": "short title",
      "reason": "why this matters",
      "config": {}
    }
  ],
  "alerts": ["alert 1", ...]
}

For email recommendations, config should include: {"template": "...", "target": "...", "subject": "...", "body_preview": "..."}
For experiment recommendations: {"name": "...", "type": "email_subject|onboarding_flow|feature_flag", "variants": [...], "hypothesis": "..."}
For blog recommendations: {"title": "...", "target_keyword": "..."}
For rule_tweak recommendations: {"rule_name": "...", "change": "..."}"""

    user_prompt = f"""Here is the weekly data for Svend. Analyze it and provide growth insights, actionable recommendations, and any alerts.

{json.dumps(prompt_data, indent=2, default=str)}"""

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Claude growth review returned invalid JSON")
        result = {
            "insights": ["Failed to parse Claude response"],
            "recommendations": [],
            "alerts": ["Review task returned non-JSON response"],
        }
    except Exception as e:
        logger.error("Claude growth review failed: %s", e)
        result = {
            "insights": [],
            "recommendations": [],
            "alerts": [f"Growth review task failed: {e}"],
        }

    report = AutopilotReport.objects.create(
        data_snapshot=prompt_data,
        insights=result.get("insights", []),
        recommendations=result.get("recommendations", []),
        alerts=result.get("alerts", []),
    )
    logger.info("Autopilot report created: %s", report.id)

    return {"report_id": str(report.id), "insights": len(result.get("insights", []))}

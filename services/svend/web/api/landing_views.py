"""Landing page views with localized pricing.

Reads CF-IPCountry from Cloudflare to show prices in local currency.
Prices match Stripe exactly — no conversion math.
"""

from django.shortcuts import render

from accounts.billing import COUNTRY_TO_REGION

# Display prices per region — must match Stripe prices exactly.
# Keyed by region code from COUNTRY_TO_REGION / REGIONAL_PRICES.
REGION_PRICING = {
    "us": {
        "currency": "USD",
        "symbol": "$",
        "symbol_after": False,
        "founder": "19",
        "pro": "49",
        "team": "99",
        "enterprise": "299",
        "pro_annual": "588",
        "has_founder": True,
    },
    "in": {
        "currency": "INR",
        "symbol": "\u20b9",
        "symbol_after": False,
        "pro": "1,499",
        "team": "3,499",
        "enterprise": "9,999",
        "pro_annual": "17,988",
        "has_founder": False,
    },
    "vn": {
        "currency": "VND",
        "symbol": "\u20ab",
        "symbol_after": True,
        "pro": "349,000",
        "team": "799,000",
        "enterprise": "2,499,000",
        "pro_annual": "4,188,000",
        "has_founder": False,
    },
    "ua": {
        "currency": "UAH",
        "symbol": "\u20b4",
        "symbol_after": False,
        "pro": "349",
        "team": "899",
        "enterprise": "2,999",
        "pro_annual": "4,188",
        "has_founder": False,
    },
    "ph": {
        "currency": "PHP",
        "symbol": "\u20b1",
        "symbol_after": False,
        "pro": "1,290",
        "team": "2,990",
        "enterprise": "8,990",
        "pro_annual": "15,480",
        "has_founder": False,
    },
    "my": {
        "currency": "MYR",
        "symbol": "RM",
        "symbol_after": False,
        "pro": "99",
        "team": "229",
        "enterprise": "699",
        "pro_annual": "1,188",
        "has_founder": False,
    },
    "id": {
        "currency": "IDR",
        "symbol": "Rp",
        "symbol_after": False,
        "pro": "249,000",
        "team": "579,000",
        "enterprise": "1,799,000",
        "pro_annual": "2,988,000",
        "has_founder": False,
    },
    "mx": {
        "currency": "MXN",
        "symbol": "MX$",
        "symbol_after": False,
        "pro": "449",
        "team": "899",
        "enterprise": "2,490",
        "pro_annual": "5,388",
        "has_founder": False,
    },
    "ae": {
        "currency": "AED",
        "symbol": "AED\u00a0",
        "symbol_after": False,
        "pro": "149",
        "team": "349",
        "enterprise": "999",
        "pro_annual": "1,788",
        "has_founder": False,
    },
    "za": {
        "currency": "ZAR",
        "symbol": "R",
        "symbol_after": False,
        "pro": "349",
        "team": "799",
        "enterprise": "2,499",
        "pro_annual": "4,188",
        "has_founder": False,
    },
    "ke": {
        "currency": "KES",
        "symbol": "KSh\u00a0",
        "symbol_after": False,
        "pro": "1,990",
        "team": "4,490",
        "enterprise": "13,990",
        "pro_annual": "23,880",
        "has_founder": False,
    },
    "ng": {
        "currency": "NGN",
        "symbol": "\u20a6",
        "symbol_after": False,
        "pro": "4,990",
        "team": "11,990",
        "enterprise": "34,990",
        "pro_annual": "59,880",
        "has_founder": False,
    },
    "br": {
        "currency": "BRL",
        "symbol": "R$",
        "symbol_after": False,
        "pro": "99",
        "team": "229",
        "enterprise": "699",
        "pro_annual": "1,188",
        "has_founder": False,
    },
    "co": {
        "currency": "COP",
        "symbol": "COP$\u00a0",
        "symbol_after": False,
        "pro": "59,900",
        "team": "139,900",
        "enterprise": "449,900",
        "pro_annual": "718,800",
        "has_founder": False,
    },
    "th": {
        "currency": "THB",
        "symbol": "\u0e3f",
        "symbol_after": False,
        "pro": "749",
        "team": "1,690",
        "enterprise": "4,990",
        "pro_annual": "8,988",
        "has_founder": False,
    },
}

DEFAULT_REGION = "us"


def get_pricing_context(request):
    """Build pricing context from Cloudflare CF-IPCountry header.

    Returns a dict with all template variables needed for localized pricing.
    """
    country = request.META.get("HTTP_CF_IPCOUNTRY", "").upper()
    if country in ("XX", "T1", ""):
        country = "US"

    region = COUNTRY_TO_REGION.get(country, DEFAULT_REGION)
    pricing = REGION_PRICING.get(region, REGION_PRICING[DEFAULT_REGION])

    sym = pricing["symbol"]
    after = pricing.get("symbol_after", False)

    def fmt(amount):
        if after:
            return f"{amount}{sym}"
        return f"{sym}{amount}"

    return {
        "region": region,
        "country": country,
        "currency": pricing["currency"],
        "currency_symbol": sym,
        "has_founder": pricing.get("has_founder", False),
        # Formatted prices
        "price_free": fmt("0"),
        "price_founder": fmt(pricing.get("founder", "19")),
        "price_pro": fmt(pricing["pro"]),
        "price_team": fmt(pricing["team"]),
        "price_enterprise": fmt(pricing["enterprise"]),
        "price_pro_annual": fmt(pricing["pro_annual"]),
        # Raw amounts (for sentences like "starting at X")
        "amount_pro": pricing["pro"],
        "amount_team": pricing["team"],
        "amount_enterprise": pricing["enterprise"],
        "amount_founder": pricing.get("founder", "19"),
    }


def landing_view(request):
    """Render landing page with localized pricing."""
    from api.models import RoadmapItem

    ctx = get_pricing_context(request)
    ctx["roadmap_items"] = RoadmapItem.objects.filter(
        is_public=True,
    ).exclude(status__in=["cancelled", "shipped"])
    return render(request, "landing.html", ctx)


def register_view(request):
    """Render register page with localized pricing context."""
    ctx = get_pricing_context(request)
    return render(request, "register.html", ctx)


def iso_qms_view(request):
    """Render ISO 9001 QMS SEO page with localized pricing."""
    ctx = get_pricing_context(request)
    return render(request, "iso_9001_qms.html", ctx)


def iso_audit_playbook_view(request):
    """Render ISO 9001 internal audit playbook with localized pricing."""
    ctx = get_pricing_context(request)
    return render(request, "iso_9001_internal_audit_playbook.html", ctx)


def svend_vs_minitab_view(request):
    """Render Svend vs Minitab comparison page with localized pricing."""
    ctx = get_pricing_context(request)
    return render(request, "svend_vs_minitab.html", ctx)


def svend_vs_jmp_view(request):
    """Render Svend vs JMP comparison page with localized pricing."""
    ctx = get_pricing_context(request)
    return render(request, "svend_vs_jmp.html", ctx)


def ci_hub_view(request):
    """Render Continuous Improvement hub page with localized pricing."""
    ctx = get_pricing_context(request)
    return render(request, "continuous_improvement.html", ctx)


def mdi_playbook_view(request):
    """Render MDI playbook with localized pricing."""
    ctx = get_pricing_context(request)
    return render(request, "mdi_playbook.html", ctx)


def hoshin_playbook_view(request):
    """Render Hoshin Kanri playbook with localized pricing."""
    ctx = get_pricing_context(request)
    return render(request, "hoshin_playbook.html", ctx)


def kaizen_playbook_view(request):
    """Render Kaizen execution guide with localized pricing."""
    ctx = get_pricing_context(request)
    return render(request, "kaizen_playbook.html", ctx)


def five_s_playbook_view(request):
    """Render 5S playbook with localized pricing."""
    ctx = get_pricing_context(request)
    return render(request, "5s_playbook.html", ctx)


def lsw_playbook_view(request):
    """Render Leadership Standard Work playbook with localized pricing."""
    ctx = get_pricing_context(request)
    return render(request, "lsw_playbook.html", ctx)


def vsm_playbook_view(request):
    """Render Value Stream Mapping playbook with localized pricing."""
    ctx = get_pricing_context(request)
    return render(request, "vsm_playbook.html", ctx)


def partnerships_view(request):
    """Render partnerships page."""
    ctx = get_pricing_context(request)
    return render(request, "partnerships.html", ctx)


def roadmap_view(request):
    """Public product roadmap showing planned and shipped features by quarter."""
    from collections import OrderedDict

    from api.models import RoadmapItem

    items = RoadmapItem.objects.filter(
        is_public=True,
    ).exclude(status="cancelled")

    # Group by quarter, sorted chronologically (newest first)
    # Parse quarter "Q2-2026" into (year, q_num) for proper sorting
    quarters = OrderedDict()
    for item in items:
        q = item.quarter
        if q not in quarters:
            quarters[q] = []
        quarters[q].append(item)

    # Sort quarters chronologically (newest first)
    def quarter_sort_key(q_str):
        try:
            parts = q_str.split("-")
            return (int(parts[1]), int(parts[0][1]))
        except (IndexError, ValueError):
            return (0, 0)

    sorted_quarters = sorted(quarters.keys(), key=quarter_sort_key, reverse=True)
    quarter_data = [(q, quarters[q]) for q in sorted_quarters]

    context = {
        "quarter_data": quarter_data,
        **get_pricing_context(request),
    }
    return render(request, "roadmap.html", context)


def education_view(request):
    """Render education partnerships page with localized alumni pricing."""
    ctx = get_pricing_context(request)

    # Compute 50% alumni discount from raw display amounts
    sym = ctx["currency_symbol"]
    pricing = REGION_PRICING.get(ctx["region"], REGION_PRICING[DEFAULT_REGION])
    after = pricing.get("symbol_after", False)

    def half(amount_str):
        """Halve a formatted number string like '1,499' or '349,000'."""
        import math

        raw = float(amount_str.replace(",", ""))
        halved = raw / 2
        # USD keeps cents ($24.50); other currencies round up to whole numbers
        if pricing["currency"] == "USD" and halved != int(halved):
            formatted = f"{halved:,.2f}"
        else:
            formatted = f"{math.ceil(halved):,}"
        if after:
            return f"{formatted}{sym}"
        return f"{sym}{formatted}"

    ctx["alumni_pro"] = half(pricing["pro"])
    ctx["alumni_team"] = half(pricing["team"])
    return render(request, "education_partnerships.html", ctx)

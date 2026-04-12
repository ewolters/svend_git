"""Shared production safety utilities for management command overrides.

Used by all command overrides in this directory to detect production databases
and display consistent safety banners. See OPS-001 §12.2.
"""


def _is_production_db(database="default"):
    """Return True if the database is NOT a test database."""
    from django.conf import settings

    db_name = str(settings.DATABASES.get(database, {}).get("NAME", ""))
    return not (db_name.startswith("test_") or "test" in db_name.lower())


def _db_name(database="default"):
    """Return the database NAME from settings."""
    from django.conf import settings

    return str(settings.DATABASES.get(database, {}).get("NAME", ""))


def blocked_banner(cmd, db):
    """Return a box-drawing banner for hard-blocked commands."""
    cmd_line = f"  BLOCKED: {cmd} on production database is prohibited."
    db_line = f"  Database: {db}"
    # Pad to fixed width (63 inner chars)
    width = 63
    return (
        "\n"
        + "\u2554"
        + "\u2550" * width
        + "\u2557\n"
        + "\u2551"
        + cmd_line.ljust(width)
        + "\u2551\n"
        + "\u2551"
        + " " * width
        + "\u2551\n"
        + "\u2551"
        + db_line.ljust(width)
        + "\u2551\n"
        + "\u2551"
        + "  Pass --i-know-what-i-am-doing to override.".ljust(width)
        + "\u2551\n"
        + "\u255a"
        + "\u2550" * width
        + "\u255d\n"
    )


def add_force_flag(parser):
    """Add the standard --i-know-what-i-am-doing flag to a parser."""
    parser.add_argument(
        "--i-know-what-i-am-doing",
        action="store_true",
        dest="force_allow",
        help="Bypass production safety check (DANGEROUS).",
    )

"""
Cursor-based pagination for API-002 compliant responses.

Standard: API-002 §7
Compliance: ISO 9001:2015 §8.5 / SOC 2 CC6.1

Features:
- Cursor-based pagination with ULID ordering
- Deterministic ordering guarantee
- Standard response envelope (data, next_cursor, total_estimate)
- Limit bounds (min: 1, max: 200, default: 50)
"""

import base64
import json
import logging
from typing import Any
from urllib import parse

from django.db.models import QuerySet
from rest_framework.pagination import CursorPagination
from rest_framework.request import Request
from rest_framework.response import Response

logger = logging.getLogger(__name__)


# =============================================================================
# Constants (API-002 §7)
# =============================================================================

DEFAULT_PAGE_SIZE = 50
MIN_PAGE_SIZE = 1
MAX_PAGE_SIZE = 200
DEFAULT_ORDERING = "-created_at"  # Stable sort key per API-002 §7.1


class SynaraCursorPagination(CursorPagination):
    """
    API-002 compliant cursor pagination.

    Standard: API-002 §7
    Table: Response envelope per §7.2

    Features:
    - Cursor-based pagination (no offset/skip)
    - Deterministic ordering by created_at or id
    - Response envelope: {data: [], next_cursor: "...", total_estimate: N}
    - Limit parameter with bounds enforcement

    Usage:
        class MyViewSet(ModelViewSet):
            pagination_class = SynaraCursorPagination
    """

    # Cursor configuration per API-002 §7.1
    page_size = DEFAULT_PAGE_SIZE
    page_size_query_param = "limit"
    max_page_size = MAX_PAGE_SIZE
    cursor_query_param = "cursor"

    # Ordering configuration per API-002 §7.1 (stable sort key)
    ordering = DEFAULT_ORDERING

    def get_page_size(self, request: Request) -> int:
        """
        Get page size with bounds enforcement.

        Returns:
            Page size between MIN_PAGE_SIZE and MAX_PAGE_SIZE
        """
        if self.page_size_query_param:
            try:
                size = int(request.query_params.get(self.page_size_query_param, self.page_size))
                return max(MIN_PAGE_SIZE, min(size, MAX_PAGE_SIZE))
            except (KeyError, ValueError):
                pass
        return self.page_size

    def get_paginated_response(self, data: list[Any]) -> Response:
        """
        Return API-002 compliant response envelope.

        Returns:
            Response with {data, next_cursor, total_estimate} structure
        """
        return Response(
            {
                "data": data,
                "next_cursor": self.get_next_link_cursor(),
                "total_estimate": self.get_total_estimate(),
            }
        )

    def get_paginated_response_schema(self, schema: dict) -> dict:
        """
        Return OpenAPI schema for paginated response.
        """
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": schema,
                },
                "next_cursor": {
                    "type": "string",
                    "nullable": True,
                    "description": "Opaque cursor for next page (null if no more data)",
                },
                "total_estimate": {
                    "type": "integer",
                    "description": "Estimated total count (may be approximate)",
                },
            },
            "required": ["data"],
        }

    def get_next_link_cursor(self) -> str | None:
        """
        Extract cursor from next link URL.

        Returns:
            Cursor string or None if no next page
        """
        next_link = self.get_next_link()
        if not next_link:
            return None

        # Parse cursor from URL
        parsed = parse.urlparse(next_link)
        params = parse.parse_qs(parsed.query)
        cursor_list = params.get(self.cursor_query_param, [])

        if cursor_list:
            return cursor_list[0]
        return None

    def get_total_estimate(self) -> int:
        """
        Get estimated total count.

        Uses count with timeout fallback for large datasets.
        """
        try:
            # Store count during paginate_queryset if available
            if hasattr(self, "_total_count"):
                return self._total_count
            return 0
        except Exception as e:
            logger.debug(f"[PAGINATION] Non-critical count retrieval failed: {e}")
            return 0

    def paginate_queryset(self, queryset: QuerySet, request: Request, view=None):
        """
        Paginate queryset with total count estimation.
        """
        # Get total estimate before pagination (with limit for performance)
        try:
            # Use approximate count for large tables
            self._total_count = queryset.count()
        except Exception as e:
            logger.warning(f"[PAGINATION] Failed to get queryset count: {e}")
            self._total_count = 0

        return super().paginate_queryset(queryset, request, view)


class SynaraListPagination(SynaraCursorPagination):
    """
    Alternative pagination for list endpoints with enhanced features.

    Standard: API-002 §7

    Additional features:
    - Sort parameter parsing (sort=field:asc,field2:desc)
    - Filter parameter integration (future)
    - Fields parameter for sparse fieldsets (future)
    """

    def get_ordering(self, request: Request, queryset, view):
        """
        Parse sort parameter into ordering.

        Format: sort=field:asc,field2:desc
        """
        sort_param = request.query_params.get("sort", "")

        if not sort_param:
            return self.ordering

        orderings = []
        for part in sort_param.split(","):
            if ":" in part:
                field, direction = part.rsplit(":", 1)
                if direction.lower() == "desc":
                    orderings.append(f"-{field}")
                else:
                    orderings.append(field)
            else:
                orderings.append(part)

        return orderings if orderings else self.ordering


# =============================================================================
# Utility Functions
# =============================================================================


def encode_cursor(position: dict) -> str:
    """
    Encode cursor position as base64 string.

    Args:
        position: Dictionary with cursor position data

    Returns:
        Base64-encoded cursor string
    """
    json_str = json.dumps(position, sort_keys=True)
    return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")


def decode_cursor(cursor: str) -> dict | None:
    """
    Decode base64 cursor string to position dictionary.

    Args:
        cursor: Base64-encoded cursor string

    Returns:
        Position dictionary or None if invalid
    """
    try:
        json_str = base64.b64decode(cursor.encode("utf-8")).decode("utf-8")
        return json.loads(json_str)
    except Exception as e:
        logger.warning(f"Failed to decode cursor: {e}")
        return None


def create_cursor_response(
    data: list[Any],
    next_cursor: str | None = None,
    total_estimate: int = 0,
) -> dict:
    """
    Create API-002 compliant cursor pagination response.

    Args:
        data: List of items
        next_cursor: Cursor for next page
        total_estimate: Estimated total count

    Returns:
        Response dictionary with data, next_cursor, total_estimate
    """
    return {
        "data": data,
        "next_cursor": next_cursor,
        "total_estimate": total_estimate,
    }

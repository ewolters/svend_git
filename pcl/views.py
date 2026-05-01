"""PCL — Process Characteristics Library API views."""

import json
import logging

logger = logging.getLogger("pcl.views")


def _json_body(request):
    """Parse request JSON body."""
    try:
        return json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return {}

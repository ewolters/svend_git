"""Learning content package — split from monolithic learn_content.py."""

from ._datasets import SHARED_DATASET  # noqa: F401
from ._registry import (
    SECTION_CONTENT,
    get_all_topics,
    get_section_content,
)  # noqa: F401

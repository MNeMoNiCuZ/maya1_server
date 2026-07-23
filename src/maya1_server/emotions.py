"""The Maya1 inline emotion-tag registry.

Maya1 supports inline emotion tags placed directly inside the spoken text, e.g.::

    Our new update <laugh> finally ships the feature you asked for.

The authoritative list is published as ``emotions.txt`` in the
``maya-research/maya1`` HuggingFace repository. It is reproduced here so the
server can validate and advertise the supported tags without a network call.

Tags that are not in this list are NOT rejected by the model - it will simply
try to interpret them - but the server can optionally warn about unknown tags
so callers catch typos early (see :func:`find_unknown_tags`).
"""

from __future__ import annotations

import re
from typing import List

# Canonical emotion tags, in the order published by Maya Research.
EMOTION_TAGS: List[str] = [
    "laugh",
    "laugh_harder",
    "sigh",
    "chuckle",
    "gasp",
    "angry",
    "excited",
    "whisper",
    "cry",
    "scream",
    "sing",
    "snort",
    "exhale",
    "gulp",
    "giggle",
    "sarcastic",
    "curious",
]

# Set form for O(1) membership checks.
EMOTION_TAG_SET = frozenset(EMOTION_TAGS)

# Matches any ``<word>`` style inline tag (letters, digits, underscore).
_TAG_PATTERN = re.compile(r"<([a-zA-Z0-9_]+)>")


def as_tags() -> List[str]:
    """Return the emotion tags in their inline ``<tag>`` form."""
    return [f"<{name}>" for name in EMOTION_TAGS]


def find_unknown_tags(text: str) -> List[str]:
    """Return any ``<...>`` tags in ``text`` that are not known emotion tags.

    This lets callers detect typos (e.g. ``<laughs>`` instead of ``<laugh>``).
    Voice descriptions are wrapped in ``<description="...">`` which is not a bare
    tag, so it is never matched here.
    """
    found = _TAG_PATTERN.findall(text)
    return [f"<{name}>" for name in found if name not in EMOTION_TAG_SET]

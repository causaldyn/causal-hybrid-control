"""MkDocs hook: render the Sphinx cross-reference roles in the docstrings as plain code.

The docstrings cross-reference with Sphinx roles (``:func:`prescribe```), which Markdown renders
as the role's name in front of a code span. Dropping the role keeps the code span, and a leading
``~`` keeps only the last dotted component, as Sphinx does.
"""

from __future__ import annotations

import re

_ROLE = re.compile(
    r":(?:py:)?(?:mod|func|class|meth|attr|data|exc|obj|const):<code>(~?)([^<]*)</code>"
)


def _plain(match: re.Match[str]) -> str:
    tilde, target = match.groups()
    return f"<code>{target.rsplit('.', 1)[-1] if tilde else target}</code>"


def on_page_content(html: str, **_: object) -> str:
    return _ROLE.sub(_plain, html)

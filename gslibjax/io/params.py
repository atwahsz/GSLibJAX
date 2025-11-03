"""Simple GSLIB parameter file parsing.

Parses the common pattern of:
- Title lines and comments
- One token per line or numbers per line used by GSLIB examples

This is a pragmatic parser for initial compatibility.
"""

from __future__ import annotations

from typing import List


def _is_comment(line: str) -> bool:
    return line.strip().startswith("#") or line.strip().startswith("//")


def read_param_file(path: str) -> List[str]:
    """Read a parameter file and return non-empty, non-comment tokens/lines.

    Args:
        path: Path to parameter file.

    Returns:
        List of stripped lines excluding blanks/comments.
    """
    out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if _is_comment(s):
                continue
            out.append(s)
    return out



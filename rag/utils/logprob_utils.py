"""Utilities for locating answer-token positions in an output token stream."""
import re
import unicodedata
from typing import List, Optional, Callable

from loguru import logger

_SNIPPET_LEN = 20
_SPECIAL_TOKEN_MARGIN = 50


def _normalize_for_match(s: str) -> str:
    """Align parsed *final_content* with raw ``decode`` strings.

    Harmony / token decode can differ from parsed text (e.g. ``\\r\\n`` vs ``\\n``,
    hyphen variants U+2011 vs ASCII, NFKC compatibility forms).
    """
    t = unicodedata.normalize("NFKC", s)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    for ch in ("\u2011", "\u2010", "\u2012", "\u2013", "\u2014", "\u2212"):
        t = t.replace(ch, "-")
    return t


def _snippet_at_decode_start(remaining: str, snippet: str) -> bool:
    """True if *snippet* appears at the start of *remaining* (after optional leading ws).

    Rejects prefix false positives such as ``No`` inside ``None`` or ``Yes`` inside
    ``Yesterday``: the character immediately after *snippet*, if any, must not be
    alphanumeric.  Compares Unicode-normalised forms so parsed text matches decode.
    """
    for candidate in (remaining, remaining.lstrip()):
        nc = _normalize_for_match(candidate)
        ns = _normalize_for_match(snippet)
        if not nc.startswith(ns):
            continue
        tail = nc[len(ns) :]
        if tail and tail[0].isalnum():
            continue
        return True
    return False


def _snippet_occurrence_in_margin(remaining: str, snippet: str) -> Optional[int]:
    """Return start index of first valid ``snippet`` in *remaining* within the margin.

    Valid means the same boundary rule as ``_snippet_at_decode_start`` at that offset.
    Uses normalised strings for ``find``.
    """
    nr = _normalize_for_match(remaining)
    ns = _normalize_for_match(snippet)
    start = 0
    while True:
        pos = nr.find(ns, start)
        if pos == -1 or pos > _SPECIAL_TOKEN_MARGIN:
            return None
        tail = nr[pos + len(ns) :]
        if not tail or not tail[0].isalnum():
            return pos
        start = pos + 1


def _last_token_index_for_snippet(
    tokens: List,
    snippet: str,
    decode_fn: Callable[[List], str],
) -> Optional[int]:
    """Return the **last** token index whose suffix decode starts with *snippet*.

    Two-phase approach — O(N + K·log N) total decode work (K = number of
    snippet occurrences, typically 1-3):

      1. Decode the full token sequence once and use ``str.find`` to locate all
         character positions where *snippet* occurs — O(N).
      2. For each hit, **binary search** over token indices using prefix decodes
         to find the exact token boundary, then verify with a small probe window.
         O(log N) prefix decodes per hit, cached across hits.
    """
    if not tokens:
        return None

    n = len(tokens)
    full_decoded = decode_fn(tokens)
    ns = _normalize_for_match(snippet)
    nf = _normalize_for_match(full_decoded)

    if not ns or not nf:
        return None

    # Phase 1: find all valid snippet positions in the normalised decoded text.
    char_positions: List[int] = []
    search_start = 0
    while True:
        pos = nf.find(ns, search_start)
        if pos == -1:
            break
        tail_start = pos + len(ns)
        if tail_start < len(nf) and nf[tail_start].isalnum():
            search_start = pos + 1
            continue
        char_positions.append(pos)
        search_start = pos + 1

    if not char_positions:
        return None

    # Phase 2: binary search for exact token index per character position.
    # Cached normalised-prefix-length avoids redundant decode calls across
    # multiple character positions.
    _prefix_cache: dict = {0: 0}

    def _norm_prefix_len(k: int) -> int:
        if k not in _prefix_cache:
            _prefix_cache[k] = len(_normalize_for_match(decode_fn(tokens[:k])))
        return _prefix_cache[k]

    _DECODE_SLICE = max(len(snippet) * 4, 60)
    _PROBE_RADIUS = 3
    candidates: List[int] = []

    for pos in char_positions:
        # Binary search: find smallest tok_idx where norm_prefix_len > pos.
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if _norm_prefix_len(mid) <= pos:
                lo = mid + 1
            else:
                hi = mid

        # Token lo-1 is the last token whose prefix doesn't overshoot pos.
        # Verify with a small probe using actual slice decodes.
        estimated = max(0, lo - 1)
        probe_lo = max(0, estimated - _PROBE_RADIUS)
        probe_hi = min(n - 1, estimated + _PROBE_RADIUS)
        for i in range(probe_lo, probe_hi + 1):
            remaining = decode_fn(tokens[i : i + _DECODE_SLICE])
            if _snippet_at_decode_start(remaining, snippet):
                candidates.append(i)

    return max(candidates) if candidates else None


def find_answer_token_position(
    output_tokens: List,
    final_content: str,
    logprob_pattern: Optional[re.Pattern],
    decode_fn: Callable[[List], str],
) -> Optional[int]:
    """Locate the token index whose logprobs should be extracted.

    Uses *logprob_pattern* on *final_content* (last match, same idea as parsing) to
    obtain the answer substring, then finds the **last** token position in the
    completion stream where that substring begins as a decoded suffix (with
    Unicode-normalised matching).  Does **not** align on a prefix of
    *final_content*, because parsed text can differ from raw ``decode`` (hyphens,
    newlines, quotes).

    Returns ``None`` when no token position can be found.
    """
    if not final_content:
        logger.warning("find_answer_token_position: final_content is empty")
        return None

    if logprob_pattern is None:
        logger.warning("find_answer_token_position called with logprob_pattern=None")
        return None

    matches = list(logprob_pattern.finditer(final_content))
    if not matches:
        logger.warning(
            "logprob_pattern {} did not match in final_content: {!r}...",
            logprob_pattern.pattern, final_content[:80],
        )
        return None
    match = matches[-1]

    target_start = match.start(1) if match.lastindex else match.end(0)
    answer_snippet = final_content[
        target_start : target_start + min(_SNIPPET_LEN, len(final_content) - target_start)
    ]
    if not answer_snippet:
        logger.warning("answer_snippet is empty after logprob_pattern match")
        return None

    tokens = list(output_tokens)
    return _last_token_index_for_snippet(tokens, answer_snippet, decode_fn)

"""Deterministic, nested typo variants."""

import hashlib
import json
import random
import re

VERSION = "internal-swap-v1"
SEED = 42

# Keep hyphenated words and contractions together so they are skipped.
TOKEN = re.compile(r"\b\w+(?:[-'’]\w+)*\b")

PROTECTED = frozenset(
    """no not nor never neither none without except unless only all any
    every each either both always sometimes often rarely before after
    above below between within more less fewer greater higher lower
    possible impossible possibly likely unlikely probably necessarily
    necessary sufficient cannot should would could might must
    increase increases increased increasing decrease decreases decreased
    decreasing causes caused prevents prevented reduces reduced
    significant significantly respectively approximately""".split()
)


def stable_seed(key: str) -> int:
    payload = json.dumps([SEED, key, VERSION], ensure_ascii=False)
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def select_ids(
    ids: list[str],
    count: int,
    label: str,
) -> list[str]:
    ordered = sorted(
        ids,
        key=lambda item: (stable_seed(f"{label}:{item}"), item),
    )
    return sorted(ordered[:count])


def typo_variants(query_id: str, text: str) -> list[dict]:
    tokens = list(TOKEN.finditer(text))
    choices = []

    for token in tokens:
        word = token.group()

        if re.fullmatch(r"[a-z]{6,}", word) is None or word in PROTECTED:
            continue

        positions = [i for i in range(1, len(word) - 2) if word[i] != word[i + 1]]

        if positions:
            choices.append((token, positions))

    seed = stable_seed(query_id)
    rng = random.Random(seed)
    rng.shuffle(choices)

    edits = []

    for token, positions in choices[:2]:
        offset = rng.choice(positions)
        before = token.group()
        letters = list(before)

        letters[offset], letters[offset + 1] = (
            letters[offset + 1],
            letters[offset],
        )

        edits.append(
            {
                "start": token.start(),
                "end": token.end(),
                "swap_at": token.start() + offset,
                "before": before,
                "after": "".join(letters),
            }
        )

    records = []
    original_words = {token.group() for token in tokens}

    for condition, count in (
        ("typo_light", 1),
        ("typo_heavy", 2),
    ):
        eligible = len(edits) >= count
        applied = edits[:count] if eligible else []

        characters = list(text)

        for edit in applied:
            i = edit["swap_at"]
            characters[i], characters[i + 1] = (
                characters[i + 1],
                characters[i],
            )

        records.append(
            {
                "variant_id": f"{query_id}::{condition}::{VERSION}",
                "parent_query_id": query_id,
                "condition": condition,
                "original_text": text,
                "text": "".join(characters),
                "seed": seed,
                "status": "pending" if eligible else "ineligible",
                "review_note": ("" if eligible else "Too few eligible words"),
                "edits": applied,
                "edit_count": len(applied),
                "edited_token_fraction": (len(applied) / len(tokens) if tokens else 0.0),
                "flagged": any(
                    edit["after"] in PROTECTED or edit["after"] in original_words
                    for edit in applied
                ),
            }
        )

    return records

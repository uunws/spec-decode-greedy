"""Draft-equivalence gate for the RQ2 retrieval-efficiency factor.

The 2x2 factorial varies retrieval *cost* (on-the-fly scan vs precomputed index)
while holding draft *quality* fixed. That only holds if every arm emits the same
draft for the same prompt. Equivalence is a property of the code *and its
settings*, not of the code alone -- a `cap_positions` that is fine for depth
drafting silently breaks width drafting -- so it must be re-checked with the
actual settings of each run, before any timing is collected.

Usage::

    prompts = sample_prompts(corpus_tokens, max_prefix=n - 1, count=512, rng=rng)
    assert_draft_equivalent(
        {"on_the_fly": scanning_drafter, "precomputed": indexed_drafter},
        prompts,
        context="cell=A k=4",
    )
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


class DraftDivergence(RuntimeError):
    """Two retrieval arms produced different drafts for the same prompt.

    This is a configuration error, not a data property: the affected cell must
    be fixed and re-run, never silently skipped or averaged in.
    """


@dataclass(frozen=True)
class EquivalenceReport:
    """Evidence that a set of arms agreed, recorded alongside the run's results."""

    arm_ids: Tuple[str, ...]
    prompts_checked: int
    corpus_tokens: int
    context: Optional[str] = None


def sample_prompts(
    corpus_tokens: Sequence[int],
    *,
    max_prefix: int,
    count: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Draw prompt suffixes from the corpus itself.

    Prompts taken from the corpus are guaranteed to hit the match paths (rather
    than the trivial no-match path), which is where the arms can disagree.
    Prefix lengths span ``1..max_prefix`` so every backoff order is exercised.
    """
    n_corpus = len(corpus_tokens)
    if n_corpus == 0 or max_prefix <= 0 or count <= 0:
        return []

    prompts: List[List[int]] = []
    for _ in range(count):
        k = int(rng.integers(1, max_prefix + 1))
        if k > n_corpus:
            continue
        end = int(rng.integers(k, n_corpus + 1))
        prompts.append(list(corpus_tokens[end - k : end]))
    return prompts


def saturated_prompts(index: Any, *, limit: int = 256) -> List[List[int]]:
    """Grams whose position list hit ``cap_positions`` -- where divergence can occur.

    An index-backed drafter can only lose a branch relative to a scanning one for a
    gram whose occurrences were actually truncated, i.e. one whose bucket is full.
    Testing exactly these grams is therefore stronger than random sampling: **if this
    returns an empty list, the cap provably never binds for this corpus**, and width
    drafting is safe without relying on sampling luck.

    Deliberately conservative: a gram occurring exactly ``cap_positions`` times was not
    truncated but is still reported, since over-testing is free and under-testing is not.
    """
    cap = getattr(index, "cap_positions", None)
    tables = getattr(index, "tables", None)
    if cap is None or not tables:
        return []

    prompts: List[List[int]] = []
    for _, table in sorted(tables.items()):
        for gram, positions in table.items():
            if len(positions) >= cap:
                prompts.append(list(gram))
                if len(prompts) >= limit:
                    return prompts
    return prompts


def equivalence_prompts(
    arms: Mapping[str, Any],
    *,
    max_prefix: int,
    count: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Prompt set for the gate: every saturated gram first, then random coverage.

    The saturated grams are the exhaustive high-risk set; the random draw is the
    catch-all for divergence causes we have not anticipated.
    """
    corpus: Sequence[int] = ()
    targeted: List[List[int]] = []
    for drafter in arms.values():
        corpus = getattr(drafter, "corpus_tokens", corpus) or corpus
        index = getattr(drafter, "index", None)
        if index is not None:
            targeted.extend(saturated_prompts(index))

    # Deduplicate: re-checking the same prompt adds cost but no evidence.
    prompts = list(targeted)
    seen = {tuple(p) for p in targeted}
    for prompt in sample_prompts(corpus, max_prefix=max_prefix, count=count, rng=rng):
        key = tuple(prompt)
        if key not in seen:
            seen.add(key)
            prompts.append(prompt)
    return prompts


def assert_draft_equivalent(
    arms: Mapping[str, Any],
    prompts: Sequence[Sequence[int]],
    *,
    context: Optional[str] = None,
) -> EquivalenceReport:
    """Verify every arm emits identical drafts, or raise ``DraftDivergence``.

    Raises on the *first* disagreement, with the offending prompt, both drafts,
    and a diagnosis of the settings that most likely caused it.
    """
    arm_ids = tuple(arms)
    if len(arm_ids) < 2:
        raise ValueError("assert_draft_equivalent needs at least two arms")

    reference_id = arm_ids[0]
    reference = arms[reference_id]

    for prompt in prompts:
        prompt_list = list(prompt)
        expected = reference.generate_draft(prompt_list)
        expected_n = getattr(reference, "last_n_used", None)

        for arm_id in arm_ids[1:]:
            actual = arms[arm_id].generate_draft(prompt_list)
            actual_n = getattr(arms[arm_id], "last_n_used", None)

            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise DraftDivergence(
                    _divergence_message(
                        reference_id, arm_id, prompt_list,
                        expected, actual, arms, context,
                    )
                )
            if expected_n != actual_n:
                raise DraftDivergence(
                    _divergence_message(
                        reference_id, arm_id, prompt_list,
                        expected, actual, arms, context,
                        extra=f"last_n_used differs: {expected_n} vs {actual_n}",
                    )
                )

    return EquivalenceReport(
        arm_ids=arm_ids,
        prompts_checked=len(prompts),
        corpus_tokens=len(getattr(reference, "corpus_tokens", ())),
        context=context,
    )


def diagnose_settings(arms: Mapping[str, Any]) -> List[str]:
    """Report settings that are known to break equivalence.

    Safe to call on its own as a pre-flight check -- it inspects configuration
    only and never runs a drafter.
    """
    notes: List[str] = []

    limits = {aid: getattr(d, "size_limit", None) for aid, d in arms.items()}
    if len({v for v in limits.values() if v is not None}) > 1:
        notes.append(f"size_limit differs across arms: {limits}. It must be identical.")

    widths = {aid: getattr(d, "num_sequences", 1) for aid, d in arms.items()}
    if len(set(widths.values())) > 1:
        notes.append(f"num_sequences differs across arms: {widths}. It must be identical.")

    for arm_id, drafter in arms.items():
        index = getattr(drafter, "index", None)
        if index is None:
            continue

        corpus_len = len(getattr(index, "corpus_tokens", ()))
        cap = getattr(index, "cap_positions", None)
        if widths.get(arm_id, 1) > 1 and cap is not None and cap < corpus_len:
            notes.append(
                f"'{arm_id}': width drafting (num_sequences>1) with cap_positions={cap} "
                f"< corpus size {corpus_len}. The index drops occurrences beyond the cap, "
                f"so this arm sees fewer distinct branches than a scanning arm. "
                f"Set cap_positions >= corpus size for width experiments."
            )

        max_k = getattr(index, "max_k", None)
        required_k = getattr(drafter, "n", 1) - 1
        if max_k is not None and max_k < required_k:
            notes.append(
                f"'{arm_id}': index max_k={max_k} < n-1={required_k}. Backoff orders above "
                f"max_k are skipped silently, so this arm matches shorter prefixes than a "
                f"scanning arm. Rebuild the index with max_k=n-1."
            )

    return notes


def _divergence_message(
    reference_id: str,
    arm_id: str,
    prompt: List[int],
    expected: torch.Tensor,
    actual: torch.Tensor,
    arms: Mapping[str, Any],
    context: Optional[str],
    extra: Optional[str] = None,
) -> str:
    lines = [
        f"Retrieval arms '{reference_id}' and '{arm_id}' produced different drafts.",
        f"  prompt      : {prompt}",
        f"  {reference_id:<12}: {expected.tolist()}",
        f"  {arm_id:<12}: {actual.tolist()}",
    ]
    if context:
        lines.insert(1, f"  context     : {context}")
    if extra:
        lines.append(f"  note        : {extra}")

    notes = diagnose_settings(arms)
    if notes:
        lines.append("Likely cause:")
        lines.extend(f"  - {note}" for note in notes)
    else:
        lines.append(
            "No known misconfiguration found -- this may be a real behavioural "
            "difference between the drafters and must be investigated before the "
            "efficiency factor can be interpreted."
        )
    return "\n".join(lines)

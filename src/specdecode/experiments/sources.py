"""Group-aware dataset sources for RQ3.

RQ3 needs one thing the rest of the repo does not: a **real** grouping key, carried
by the data, that many requests share. The pilot ran without one -- it used the
dataset name as the group -- and the result was a null: group scope and global scope
scored 0.1367 and 0.1366, because history from an unrelated dataset is inert rather
than harmful. Nothing was learned about scoping, only about mixing corpora.

So every source here is chosen for its key first and its content second:

    wildchat   hashed_ip          one real user's earlier conversations
    swetraj    instance_id        one agent debugging session, step by step
    codesearch repository_name    one codebase's idioms
    spider     db_id              one database schema
    squad      title              one Wikipedia article's questions
    samsum     -                  no key: the deliberate negative case

``src/specdecode/datasets/`` is excluded from ruff, pyright and coverage because it
holds RCA-era loaders. These are load-bearing experiment code, so they live here
instead and stay inside the quality gates.

Each source yields :class:`Sample` in a stable order. A source never tokenizes --
the caller owns the tokenizer -- and never decides what is history and what is a
request, which is :mod:`specdecode.experiments.rq3_workloads`'s job.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

# HF's `datasets` is an optional heavy dependency; import it only when a source runs.


@dataclass(frozen=True)
class Sample:
    """One unit of work as the data itself defines it.

    ``doc_text`` is what a real system would already have in hand when the request
    arrives (the prompt, the surrounding code, the paragraph). ``target_text`` is
    what the target model must produce. Keeping them separate is what lets
    request-local scoping be honest: it retains the prompt, never the answer.
    """

    group_id: str
    doc_text: str
    target_text: str


@dataclass(frozen=True)
class SourceSpec:
    """Everything the runner needs to describe a source without loading it."""

    name: str
    hf_id: str
    group_field: str
    grouping: str  # "natural" or "none"
    note: str


SPECS: Dict[str, SourceSpec] = {
    "wildchat": SourceSpec(
        "wildchat", "allenai/WildChat-1M", "hashed_ip", "natural",
        "one real user's earlier turns; the user-based history case",
    ),
    "swetraj": SourceSpec(
        "swetraj", "nebius/SWE-agent-trajectories", "instance_id", "natural",
        "agentic trace; the SuffixDecoding and DAS case",
    ),
    "codesearch": SourceSpec(
        "codesearch", "code-search-net/code_search_net", "repository_name", "natural",
        "one repository's idioms; lowest-entropy natural text",
    ),
    "spider": SourceSpec(
        "spider", "xlangai/spider", "db_id", "natural",
        "structured output over a shared schema",
    ),
    "squad": SourceSpec(
        "squad", "rajpurkar/squad", "title", "natural",
        "extractive QA; the answer sits inside the local paragraph",
    ),
    "samsum": SourceSpec(
        "samsum", "knkarthick/samsum", "", "none",
        "no group key: the negative case, global vs local only",
    ),
    "cnn": SourceSpec(
        "cnn", "abisee/cnn_dailymail", "", "none",
        "news summarization; highlights copy phrases out of the article",
    ),
    "xsum": SourceSpec(
        "xsum", "EdinburghNLP/xsum", "", "none",
        "abstractive one-line summaries; rewrites rather than copies",
    ),
    "gsm8k": SourceSpec(
        "gsm8k", "openai/gsm8k", "", "none",
        "grade-school maths; worked solutions share a rigid step format",
    ),
    "dolly": SourceSpec(
        "dolly", "databricks/databricks-dolly-15k", "", "none",
        "open-ended instructions; answers share little beyond ordinary English",
    ),
}

ALL_SOURCES: Sequence[str] = (
    "wildchat", "swetraj", "codesearch", "spider", "squad", "samsum",
    "cnn", "xsum", "gsm8k", "dolly",
)


def _rows(hf_id: str, config: str = "", split: str = "train") -> Iterator[Dict[str, Any]]:
    """Stream a split as plain dicts.

    Everything below indexes rows by name, and Hugging Face row objects are typed
    loosely enough that a static checker cannot follow them. Normalising to ``dict``
    at the boundary keeps the loaders themselves checkable.
    """
    from datasets import load_dataset

    if config:
        dataset = load_dataset(hf_id, config, split=split, streaming=True)
    else:
        dataset = load_dataset(hf_id, split=split, streaming=True)
    for row in dataset:
        yield dict(row)  # type: ignore[arg-type]


def _short(value: str) -> str:
    """Stable short group id -- raw hashed IPs and repo paths make unreadable keys."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


MODEL_ROLES = ("assistant", "ai")
INPUT_ROLES = ("user", "tool", "system", "human")


def _turns(
    conversation: Sequence[Dict[str, Any]], content_field: str = "content"
) -> Iterator[Tuple[str, str]]:
    """(input turn, model turn) pairs from a role-tagged message list.

    Field and role names vary by dataset and are not guessable: WildChat uses
    ``content`` with ``assistant``, the SWE-agent trajectories use ``text`` with
    ``ai``. Both vocabularies are accepted rather than assumed, because a role name
    that never matches yields zero samples silently, which is what happened on the
    first attempt at the trajectory source.
    """
    pending: Optional[str] = None
    for message in conversation:
        role = str(message.get("role", ""))
        content = str(message.get(content_field) or "")
        if role in INPUT_ROLES:
            pending = content
        elif role in MODEL_ROLES and pending is not None:
            yield pending, content
            pending = None


def iter_wildchat(limit: int, *, min_turns: int = 4, language: str = "English") -> Iterator[Sample]:
    """Real user sessions, grouped by ``hashed_ip``.

    Only users with several conversations are kept: a group of one is
    indistinguishable from request-local and would quietly turn the middle rung of
    the scope ladder into a duplicate of the bottom one. English-only, because a
    mixed-script datastore inflates support with tokens no request will ever emit.
    """
    rows = _rows(SPECS["wildchat"].hf_id)
    seen: Dict[str, int] = {}
    produced = 0
    for row in rows:
        if produced >= limit:
            return
        if str(row.get("language")) != language or int(row.get("turn", 0)) < min_turns:
            continue
        group = _short(str(row.get("hashed_ip", "")))
        seen[group] = seen.get(group, 0) + 1
        for prompt, answer in _turns(row.get("conversation") or []):
            if not prompt.strip() or not answer.strip():
                continue
            yield Sample(group_id=f"user:{group}", doc_text=prompt, target_text=answer)
            produced += 1
            if produced >= limit:
                return


def iter_swetraj(limit: int) -> Iterator[Sample]:
    """Agentic traces, grouped by trajectory (one debugging session).

    Each model turn is a request and the tool/user turn before it is the local
    context, which is how an agent loop is actually served: the model emits an
    action, the harness answers, the model emits the next. Repetition across the
    steps of one session is the effect SuffixDecoding reports.
    """
    rows = _rows(SPECS["swetraj"].hf_id)
    produced = 0
    for row in rows:
        if produced >= limit:
            return
        # The group is the trajectory, not the repository. Repository was the first
        # choice because it is the larger group, but the corpus is SWE-bench-extra:
        # repositories barely repeat (21 distinct repos in the first 25 instances),
        # so a repo group never accumulates enough turns. A trajectory carries ~27
        # model turns, which is the per-problem history DAS describes and the
        # within-session repetition SuffixDecoding reports.
        instance = str(row.get("instance_id") or "unknown")
        for prompt, answer in _turns(row.get("trajectory") or [], content_field="text"):
            if not answer.strip():
                continue
            yield Sample(group_id=f"traj:{instance}", doc_text=prompt, target_text=answer)
            produced += 1
            if produced >= limit:
                return


def iter_codesearch(limit: int, *, language: str = "python") -> Iterator[Sample]:
    """Functions grouped by repository.

    The documentation string is the input and the function body is the target, so
    the task is "write this function" rather than "copy it back". Within a
    repository, helper names, imports and error-handling shapes repeat, which is
    the relevance that group scoping is supposed to capture.
    """
    rows = _rows(SPECS["codesearch"].hf_id, config=language)
    produced = 0
    for row in rows:
        if produced >= limit:
            return
        body = str(row.get("func_code_string") or row.get("whole_func_string") or "")
        doc = str(row.get("func_documentation_string") or "")
        repo = str(row.get("repository_name") or "unknown")
        if not body.strip():
            continue
        yield Sample(group_id=f"repo:{repo}", doc_text=doc or body[:200], target_text=body)
        produced += 1


def iter_spider(limit: int) -> Iterator[Sample]:
    """SQL grouped by database.

    Queries over one schema reuse table and column names almost verbatim, so this
    anchors the low-entropy end of the Structure axis. Targets are short (~30
    tokens), which the runner accounts for by reporting per-token acceptance
    alongside accepted length.
    """
    rows = _rows(SPECS["spider"].hf_id)
    produced = 0
    for row in rows:
        if produced >= limit:
            return
        query = str(row.get("query") or "")
        question = str(row.get("question") or "")
        db = str(row.get("db_id") or "unknown")
        if not query.strip():
            continue
        yield Sample(group_id=f"db:{db}", doc_text=question, target_text=query)
        produced += 1


def iter_squad(limit: int) -> Iterator[Sample]:
    """Extractive QA grouped by Wikipedia article title.

    ``datasets/squad.py`` drops ``title`` at the ``sample[...]`` line, which is why
    the loader is repeated here rather than reused. The target is the answer alone
    -- roughly three tokens -- so this source measures per-token acceptance and a
    position on the plane, not a speedup.
    """
    rows = _rows(SPECS["squad"].hf_id)
    produced = 0
    for row in rows:
        if produced >= limit:
            return
        answers = row.get("answers") or {}
        texts = answers.get("text") or []
        if not texts:
            continue
        yield Sample(
            group_id=f"title:{row.get('title', 'unknown')}",
            doc_text=str(row.get("context") or ""),
            target_text=str(texts[0]),
        )
        produced += 1


def iter_samsum(limit: int) -> Iterator[Sample]:
    """Dialogue summarization with **no** group key.

    Every sample gets its own group, so group scope collapses onto request-local by
    construction. That is the point: it is the control that shows what the ladder
    looks like when the data has no grouping to exploit, and the pilot already found
    request-local losing badly here (0.0264 against 0.0979 for a wide scope).
    """
    rows = _rows(SPECS["samsum"].hf_id)
    produced = 0
    for i, row in enumerate(rows):
        if produced >= limit:
            return
        summary = str(row.get("summary") or "")
        dialogue = str(row.get("dialogue") or "")
        if not summary.strip():
            continue
        yield Sample(group_id=f"none:{i}", doc_text=dialogue, target_text=summary)
        produced += 1


def iter_cnn(limit: int) -> Iterator[Sample]:
    """News articles paired with their highlight bullets.

    Highlights lift phrases out of the article almost verbatim, so this is the
    high end of what summarization offers a retrieval drafter. RQ2 never puts a
    target's own article in the datastore, so what remains is the boilerplate a
    newsroom repeats across stories, not a planted copy.
    """
    rows = _rows(SPECS["cnn"].hf_id, config="3.0.0")
    produced = 0
    for i, row in enumerate(rows):
        if produced >= limit:
            return
        summary = str(row.get("highlights") or "")
        article = str(row.get("article") or "")
        if not summary.strip() or not article.strip():
            continue
        yield Sample(group_id=f"none:{i}", doc_text=article, target_text=summary)
        produced += 1


def iter_xsum(limit: int) -> Iterator[Sample]:
    """The abstractive counterpart to CNN: one sentence, rewritten not copied.

    Included as an a-priori low-speculatability candidate. It stays in the
    candidate set whatever the measurement says -- dropping a dataset because its
    numbers came out inconvenient is exactly the circularity RQ2 has to avoid.
    """
    rows = _rows(SPECS["xsum"].hf_id)
    produced = 0
    for i, row in enumerate(rows):
        if produced >= limit:
            return
        summary = str(row.get("summary") or "")
        document = str(row.get("document") or "")
        if not summary.strip() or not document.strip():
            continue
        yield Sample(group_id=f"none:{i}", doc_text=document, target_text=summary)
        produced += 1


def iter_gsm8k(limit: int) -> Iterator[Sample]:
    """Worked arithmetic solutions, which follow a near-templated step format.

    Added to populate the structured end of the candidate set. Solutions carry
    calculator annotations and a fixed ``#### <answer>`` terminator, so successive
    solutions repeat their scaffolding even when the numbers differ.
    """
    rows = _rows(SPECS["gsm8k"].hf_id, config="main")
    produced = 0
    for i, row in enumerate(rows):
        if produced >= limit:
            return
        answer = str(row.get("answer") or "")
        question = str(row.get("question") or "")
        if not answer.strip():
            continue
        yield Sample(group_id=f"none:{i}", doc_text=question, target_text=answer)
        produced += 1


def iter_dolly(limit: int) -> Iterator[Sample]:
    """Open-ended instruction following: the a-priori low-speculatability candidate.

    Successive answers are about unrelated things and share only ordinary English,
    which is what the bottom-left of the plane is supposed to look like. It stays in
    the set whatever it measures.
    """
    rows = _rows(SPECS["dolly"].hf_id)
    produced = 0
    for i, row in enumerate(rows):
        if produced >= limit:
            return
        response = str(row.get("response") or "")
        context = str(row.get("context") or "") or str(row.get("instruction") or "")
        if not response.strip():
            continue
        yield Sample(group_id=f"none:{i}", doc_text=context, target_text=response)
        produced += 1


ITERATORS: Dict[str, Callable[[int], Iterator[Sample]]] = {
    "wildchat": iter_wildchat,
    "swetraj": iter_swetraj,
    "codesearch": iter_codesearch,
    "spider": iter_spider,
    "squad": iter_squad,
    "samsum": iter_samsum,
    "cnn": iter_cnn,
    "xsum": iter_xsum,
    "gsm8k": iter_gsm8k,
    "dolly": iter_dolly,
}


def load_samples(source: str, limit: int) -> List[Sample]:
    """Materialize ``limit`` samples from ``source``, in dataset order."""
    if source not in ITERATORS:
        raise ValueError(f"unknown source {source!r}; known: {sorted(ITERATORS)}")
    return list(ITERATORS[source](limit))


def group_histogram(samples: Sequence[Sample]) -> Dict[str, int]:
    """Samples per group -- the runner refuses to run a source whose groups are singletons."""
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.group_id] = counts.get(sample.group_id, 0) + 1
    return counts

"""
Generic multilingual Wikipedia loader (Hugging Face ``wikimedia/wikipedia``).

A teammate runs the whole RCA on a *different* language by changing ONLY the
language code — e.g. ``lo`` (Lao), ``my`` (Burmese), ``ar`` (Arabic),
``ru`` (Russian), ``uk`` (Ukrainian). Everything downstream is language-agnostic.

    from specdecode.datasets.wiki import load_articles
    texts = load_articles("lo")            # all Lao articles
    texts = load_articles("ru", max_chars=50_000_000)  # cap huge languages
"""

from typing import Callable, Dict, List, Optional, Tuple

from datasets import load_dataset

DEFAULT_DATE = "20231101"
_CACHE = {}

TokenEncoder = Callable[[str], List[int]]
ProgressCallback = Callable[[Dict[str, int]], None]


def split_tail_holdout(
    articles: List[str],
    corpus_count: int = 100,
    target_count: int = 100,
) -> Tuple[List[str], List[str]]:
    """Split leading corpus articles from a disjoint tail target holdout.

    The corpus is selected as ``articles[:corpus_count]`` and the target holdout
    as ``articles[-target_count:]``. This keeps the target articles out of the
    corpus while allowing multiple target articles to be concatenated later.
    """
    if corpus_count <= 0:
        raise ValueError("corpus_count must be positive")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if corpus_count + target_count > len(articles):
        raise ValueError(
            "corpus_count + target_count must not exceed the number of articles"
        )

    return articles[:corpus_count], articles[-target_count:]


def concatenate_articles(articles: List[str], separator: str = "\n\n") -> str:
    """Concatenate article texts into one target text."""
    return separator.join(articles)


def stream_tail_holdout_tokens(
    lang: str,
    encode: TokenEncoder,
    corpus_token_limit: int,
    target_article_count: Optional[int] = 100,
    target_token_limit: Optional[int] = None,
    date: str = DEFAULT_DATE,
    target_separator: str = "\n\n",
    progress_callback: Optional[ProgressCallback] = None,
    progress_every: int = 1_000,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Stream a Wikipedia split with a token-bounded corpus and tail holdout.

    The corpus is streamed from the front only until ``corpus_token_limit`` is
    reached. The target is loaded with a negative split such as ``train[-100:]``;
    when ``target_token_limit`` is provided, that tail window doubles until its
    concatenated tokens reach the requested limit.

    ``progress_callback`` receives counters such as ``articles_seen`` and
    ``corpus_tokens`` every ``progress_every`` articles and once at completion.
    """
    if corpus_token_limit <= 0:
        raise ValueError("corpus_token_limit must be positive")
    if target_article_count is not None and target_article_count <= 0:
        raise ValueError("target_article_count must be positive")
    if target_token_limit is not None and target_token_limit <= 0:
        raise ValueError("target_token_limit must be positive when provided")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    config = available_config(lang, date)
    corpus_tokens: List[int] = []
    articles_seen = 0
    corpus_articles_encoded = 0

    def add_corpus_tokens(article_tokens: List[int]) -> None:
        nonlocal corpus_articles_encoded
        if len(corpus_tokens) >= corpus_token_limit:
            return
        remaining = corpus_token_limit - len(corpus_tokens)
        corpus_tokens.extend(article_tokens[:remaining])
        corpus_articles_encoded += 1

    def emit_progress(target_tokens: int = 0, target_articles: int = 0) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "articles_seen": articles_seen,
                "corpus_articles_encoded": corpus_articles_encoded,
                "target_articles_buffered": target_articles,
                "corpus_tokens": len(corpus_tokens),
                "target_tokens": target_tokens,
            }
        )

    dataset = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    for row in dataset:
        articles_seen += 1
        add_corpus_tokens(encode(row["text"]))
        if articles_seen % progress_every == 0:
            emit_progress()
        if len(corpus_tokens) >= corpus_token_limit:
            break

    if not corpus_tokens:
        raise ValueError(
            "Wikipedia stream did not contain enough articles to build the requested split"
        )

    tail_window = target_article_count or 100
    target_articles: List[str] = []
    target_tokens: List[int] = []
    while True:
        tail_dataset = load_dataset(
            "wikimedia/wikipedia",
            config,
            split=f"train[-{tail_window}:]",
            streaming=False,
        )
        target_articles = [row["text"] for row in tail_dataset]
        target_text = concatenate_articles(target_articles, separator=target_separator)
        target_tokens = encode(target_text)

        enough_target = target_token_limit is None or len(target_tokens) >= target_token_limit
        if enough_target:
            if target_token_limit is not None:
                target_tokens = target_tokens[-target_token_limit:]
            break
        if len(target_articles) < tail_window:
            raise ValueError("Wikipedia tail does not contain enough target tokens")
        tail_window *= 2

    stats = {
        "articles_seen": articles_seen,
        "corpus_articles_encoded": corpus_articles_encoded,
        "target_articles_available": len(target_articles),
        "target_articles_encoded": len(target_articles),
        "target_window_articles": tail_window,
        "corpus_tokens": len(corpus_tokens),
        "target_tokens": len(target_tokens),
    }
    emit_progress(target_tokens=len(target_tokens), target_articles=len(target_articles))
    return corpus_tokens, target_tokens, stats


def available_config(lang: str, date: str = DEFAULT_DATE) -> str:
    """Config/subset name used by wikimedia/wikipedia, e.g. '20231101.lo'."""
    return f"{date}.{lang}"


def load_articles(
    lang: str,
    date: str = DEFAULT_DATE,
    max_chars: Optional[int] = None,
    streaming: bool = False,
) -> List[str]:
    """
    Return a list of plain-text Wikipedia articles for ``lang``.

    Args:
        lang:      ISO language code (lo, my, ar, ru, uk, ...).
        date:      Wikipedia dump snapshot (default 20231101).
        max_chars: stop once this many characters have been collected
                   (use for very large languages; None = load everything).
        streaming: stream instead of materialising the full split first
                   (recommended together with max_chars for big languages).
    """
    config = available_config(lang, date)
    cache_key = (config, max_chars, streaming)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=streaming)

    texts: List[str] = []
    total = 0
    for row in ds:
        t = row["text"]
        texts.append(t)
        total += len(t)
        if max_chars is not None and total >= max_chars:
            break

    _CACHE[cache_key] = texts
    return texts

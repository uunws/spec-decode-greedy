from specdecode.datasets.wiki import (
    concatenate_articles,
    split_tail_holdout,
    stream_tail_holdout_tokens,
)


def test_split_tail_holdout_keeps_corpus_and_target_disjoint() -> None:
    articles = [f"article-{index}" for index in range(10)]

    corpus, target = split_tail_holdout(articles, corpus_count=3, target_count=2)

    assert corpus == ["article-0", "article-1", "article-2"]
    assert target == ["article-8", "article-9"]
    assert set(corpus).isdisjoint(target)


def test_concatenate_articles_joins_multiple_targets() -> None:
    assert concatenate_articles(["first", "second"]) == "first\n\nsecond"


def test_split_tail_holdout_rejects_overlapping_counts() -> None:
    articles = ["one", "two", "three"]

    try:
        split_tail_holdout(articles, corpus_count=2, target_count=2)
    except ValueError as error:
        assert "must not exceed" in str(error)
    else:
        raise AssertionError("expected ValueError for overlapping split counts")


def test_stream_tail_holdout_tokens_tracks_corpus_progress(monkeypatch) -> None:
    rows = [{"text": text} for text in ["a", "bb", "ccc", "dddd", "eeeee"]]

    def fake_load_dataset(*args, **kwargs):
        split = kwargs["split"]
        if split.startswith("train[-"):
            count = int(split[len("train[-") : -2])
            return rows[-count:]
        return rows

    monkeypatch.setattr(
        "specdecode.datasets.wiki.load_dataset",
        fake_load_dataset,
    )
    progress = []

    corpus, target, stats = stream_tail_holdout_tokens(
        lang="en",
        encode=lambda text: list(text),
        corpus_token_limit=5,
        target_article_count=2,
        progress_callback=progress.append,
        progress_every=1,
    )

    assert len(corpus) == 5
    assert target == list("dddd\n\neeeee")
    assert stats["articles_seen"] == 3
    assert stats["corpus_tokens"] == 5
    assert stats["target_tokens"] == 11
    assert progress[-1]["corpus_tokens"] == 5


def test_stream_tail_holdout_tokens_expands_target_until_token_limit(monkeypatch) -> None:
    rows = [{"text": text} for text in ["a", "bb", "ccc", "dddd", "eeeee"]]

    def fake_load_dataset(*args, **kwargs):
        split = kwargs["split"]
        if split.startswith("train[-"):
            count = int(split[len("train[-") : -2])
            return rows[-count:]
        return rows

    monkeypatch.setattr(
        "specdecode.datasets.wiki.load_dataset",
        fake_load_dataset,
    )

    corpus, target, stats = stream_tail_holdout_tokens(
        lang="en",
        encode=lambda text: list(text),
        corpus_token_limit=6,
        target_article_count=1,
        target_token_limit=11,
    )

    assert len(corpus) == 6
    assert target == list("dddd\n\neeeee")
    assert stats["target_articles_available"] == 2
    assert stats["target_articles_encoded"] == 2
    assert stats["target_tokens"] == 11

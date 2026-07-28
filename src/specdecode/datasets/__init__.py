from .wiki_demo import load as load_wiki_demo
from .squad import load as load_squad
from .samsum import load as load_samsum
from .xsum import load as load_xsum
from .cnn_dailymail import load as load_cnn_dailymail
from .wiki import (
    concatenate_articles,
    load_articles,
    split_tail_holdout,
    stream_tail_holdout_tokens,
)

REGISTRY = {
    "wiki_demo": load_wiki_demo,
    "squad": load_squad,
    "samsum": load_samsum,
    "xsum": load_xsum,
    "cnn_dailymail": load_cnn_dailymail,
}


def get_dataset(name: str, index: int = 0) -> tuple:
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](index=index)

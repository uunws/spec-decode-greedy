from .wiki_demo import load as load_wiki_demo
from .squad import load as load_squad
from .samsum import load as load_samsum
from .xsum import load as load_xsum
from .thai_wiki import load as load_thai_wiki
from .wongnai import load as load_wongnai

REGISTRY = {
    "wiki_demo": load_wiki_demo,
    "squad": load_squad,
    "samsum": load_samsum,
    "xsum": load_xsum,
    "thai_wiki": load_thai_wiki,
    "wongnai": load_wongnai,
}


def get_dataset(name: str, index: int = 0) -> tuple:
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](index=index)

"""
Wongnai Reviews Dataset loader.
Corpus: Concatenated text from multiple reviews (to form a knowledge base).
Target: The text of the requested review index.
"""

from datasets import load_dataset

_CACHE = None
_CORPUS_CACHE = None

def load(index: int = 0) -> tuple:
    global _CACHE, _CORPUS_CACHE
    
    if _CACHE is None:
        print("Fetching Wongnai reviews dataset...")
        _CACHE = load_dataset("wongnai_reviews", split="train")
        
        print("Building Wongnai corpus from first 100 reviews...")
        corpus_texts = []
        for i in range(100):
            corpus_texts.append(_CACHE[i]["review_body"])
        _CORPUS_CACHE = " ".join(corpus_texts)
    
    target_idx = 100 + index
    if target_idx >= len(_CACHE):
        target_idx = index
        
    target = _CACHE[target_idx]["review_body"]
    
    return _CORPUS_CACHE, target

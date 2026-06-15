"""
Thai Wiki Dataset loader (using official Wikimedia Parquet dump).
Corpus: The first half of the wikipedia article.
Target: The second half of the wikipedia article.
"""

from datasets import load_dataset
import itertools

def load(index: int = 0) -> tuple:
    print(f"Streaming Thai Wiki (Wikimedia) sample {index} from Hugging Face...")
    # Using the official wikimedia dump which is in parquet format (no custom script needed)
    dataset = load_dataset("wikimedia/wikipedia", "20231101.th", split="train", streaming=True)
    
    dataset_iter = iter(dataset)
    try:
        sample = next(itertools.islice(dataset_iter, index, None))
    except StopIteration:
        raise ValueError(f"Index {index} out of bounds for Thai Wiki dataset")
        
    text = sample["text"]
    
    # Split text into corpus and target
    mid = len(text) // 2
    corpus = text[:mid]
    target = text[mid:]
    
    return corpus, target

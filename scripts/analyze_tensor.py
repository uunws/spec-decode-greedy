import torch
from transformers import AutoTokenizer
from src.datasets import get_dataset
from src.simulator import TensorNGramDrafter, TensorGreedyVerifier, PlaybackMetrics

def run_playback(tokenizer, corpus_tokens, target_text, S, T, n=3):
    drafter = TensorNGramDrafter(corpus_tokens, n=n, num_sequences=S, draft_depth=T)
    verifier = TensorGreedyVerifier()
    
    complete_tokens = tokenizer.encode(target_text)
    metrics = PlaybackMetrics()
    metrics.normal_steps = len(complete_tokens) - 1
    
    current_prefix = [complete_tokens[0]]
    step_idx = 0
    while len(current_prefix) < len(complete_tokens):
        draft_tokens = drafter.generate_draft(current_prefix)
        # verifier returns chosen_sequence, accepted_count
        res = verifier.verify(draft_tokens, current_prefix, complete_tokens)
        
        accepted_tokens = res["accepted_tokens"]
        accepted_count = res["accepted_count"]
        rejected_count = res["rejected_count"]
        
        if not accepted_tokens:
            next_idx = len(current_prefix)
            if next_idx < len(complete_tokens):
                current_prefix.append(complete_tokens[next_idx])
            metrics.record_step(0, 0, draft_size=T, step_idx=step_idx, context_ids=current_prefix, draft_ids=[], complete_tokens=complete_tokens)
        else:
            current_prefix.extend(accepted_tokens)
            metrics.record_step(accepted_count, rejected_count, draft_size=T, step_idx=step_idx)
            
        step_idx += 1
        
    metrics.total_tokens_generated = len(current_prefix)
    return metrics

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # 1. Subword Tokenization Example
    words = ["มหานคร", "ผลิตภัณฑ์", "สมาร์ทโฟน"]
    print("=== Subword Tokenization Example ===")
    for word in words:
        tokens = tokenizer.encode(word)
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        print(f"Word: {word}")
        print(f"Tokens: {decoded_tokens}")
        print(f"Token IDs: {tokens}\n")

    # 2. Benchmark Depth vs Width
    datasets = ["wongnai", "thai_wiki"]
    for ds in datasets:
        corpus_text, target_text = get_dataset(ds, index=0)
        corpus_tokens = tokenizer.encode(corpus_text)
        target_tokens = tokenizer.encode(target_text)
        
        print(f"=== Dataset: {ds} ===")
        print(f"Corpus size: {len(corpus_tokens)} tokens")
        print(f"Target size: {len(target_tokens)} tokens")
        print(f"Tokens to analyze (Target size - 1): {len(target_tokens) - 1} steps in baseline\n")
        
        # Depth: S=1, T=6
        m_depth = run_playback(tokenizer, corpus_tokens, target_text, S=1, T=6)
        print(f"  [Depth-Drafting] S=1, T=6 (Budget 6) | Steps: {m_depth.speculative_steps} | Speedup: {m_depth.speedup_ratio:.2f}x | Avg Accept: {m_depth.average_accepted_per_step:.2f}")
        
        # Width: S=2, T=3
        m_width1 = run_playback(tokenizer, corpus_tokens, target_text, S=2, T=3)
        print(f"  [Width-Drafting] S=2, T=3 (Budget 6) | Steps: {m_width1.speculative_steps} | Speedup: {m_width1.speedup_ratio:.2f}x | Avg Accept: {m_width1.average_accepted_per_step:.2f}")

        # Width: S=3, T=2
        m_width2 = run_playback(tokenizer, corpus_tokens, target_text, S=3, T=2)
        print(f"  [Width-Drafting] S=3, T=2 (Budget 6) | Steps: {m_width2.speculative_steps} | Speedup: {m_width2.speedup_ratio:.2f}x | Avg Accept: {m_width2.average_accepted_per_step:.2f}")
        print()

if __name__ == '__main__':
    main()

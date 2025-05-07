import json
import numpy as np
import time
from cs336_basics.simple_tokenizer import Tokenizer

# === CONFIG ===
TINYSTORIES_JSONL = "/data/a1-basics/TinyStoriesV2-GPT4-train.jsonl"
OWT_TXT = "/data/a1-basics/owt_train.txt"
TINYSTORIES_VOCAB = "/path/to/tinystories_vocab.pkl"  # TODO: update path
TINYSTORIES_MERGES = "/path/to/tinystories_merges.pkl"  # TODO: update path
OWT_VOCAB = "/path/to/owt_vocab.pkl"  # TODO: update path
OWT_MERGES = "/path/to/owt_merges.pkl"  # TODO: update path
SPECIAL_TOKENS = ["<|endoftext|>"]

# === 1. Sample 10 documents from each dataset ===
def sample_tinystories(n=10):
    docs = []
    with open(TINYSTORIES_JSONL, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            docs.append(obj["text"] if "text" in obj else obj)
    return docs

def sample_owt(n=10):
    docs = []
    with open(OWT_TXT, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            docs.append(line.strip())
    return docs

tinystories_samples = sample_tinystories(10)
owt_samples = sample_owt(10)

# === 2. Load both tokenizers ===
tinystories_tokenizer = Tokenizer.from_files(TINYSTORIES_VOCAB, TINYSTORIES_MERGES, special_tokens=SPECIAL_TOKENS)
owt_tokenizer = Tokenizer.from_files(OWT_VOCAB, OWT_MERGES, special_tokens=SPECIAL_TOKENS)

# === 3. Encode and compare ===
def analyze(samples, tokenizer, tokenizer_name, dataset_name):
    all_token_ids = []
    total_bytes = 0
    total_tokens = 0
    start = time.time()
    for doc in samples:
        token_ids = tokenizer.encode(doc)
        all_token_ids.append(np.array(token_ids, dtype=np.uint16))
        total_bytes += len(doc.encode('utf-8'))
        total_tokens += len(token_ids)
    elapsed = time.time() - start
    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else float('inf')
    throughput = total_bytes / elapsed if elapsed > 0 else 0
    print(f"{tokenizer_name} on {dataset_name}:")
    print(f"  Compression ratio (bytes/token): {compression_ratio:.2f}")
    print(f"  Throughput: {throughput:.2f} bytes/sec")
    print(f"  Total tokens: {total_tokens}, Total bytes: {total_bytes}, Time: {elapsed:.2f}s")
    return all_token_ids, compression_ratio, throughput

# TinyStories tokenizer on TinyStories
ts_ts_ids, ts_ts_cr, ts_ts_tp = analyze(tinystories_samples, tinystories_tokenizer, "TinyStoriesTokenizer", "TinyStories")
# OWT tokenizer on OWT
owt_owt_ids, owt_owt_cr, owt_owt_tp = analyze(owt_samples, owt_tokenizer, "OWTTokenizer", "OpenWebText")
# TinyStories tokenizer on OWT
ts_owt_ids, ts_owt_cr, ts_owt_tp = analyze(owt_samples, tinystories_tokenizer, "TinyStoriesTokenizer", "OpenWebText")

# === 4. Save tokenized outputs as uint16 numpy arrays ===
np.save("tinystories_sample_tokenized.npy", np.concatenate(ts_ts_ids))
np.save("owt_sample_tokenized.npy", np.concatenate(owt_owt_ids))
np.save("owt_sample_tokenized_by_tinystories.npy", np.concatenate(ts_owt_ids))

# === 5. Estimate time to tokenize the Pile (825GB) ===
PILE_SIZE_BYTES = 825 * 1024**3
est_seconds = PILE_SIZE_BYTES / owt_owt_tp if owt_owt_tp > 0 else float('inf')
print(f"Estimated time to tokenize the Pile (825GB) with OWT tokenizer: {est_seconds/3600:.2f} hours")

# === 6. Why uint16? ===
print("\nWhy uint16? Because most vocabularies are <65536 tokens, so each token ID fits in 2 bytes, saving space over uint32.") 
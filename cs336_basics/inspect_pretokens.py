import numpy as np

pretokens_path = "/data/c-kaitwang/tinystories_pretokens.npy"
tokens = np.load(pretokens_path, mmap_mode="r")

print(f"Pretokenized data shape: {tokens.shape}")
print(f"Pretokenized data dtype: {tokens.dtype}")
print(f"Min token id: {tokens.min()}")
print(f"Max token id: {tokens.max()}")

# If the range is small, print unique values
if tokens.max() - tokens.min() < 100:
    print(f"Unique token ids: {np.unique(tokens)}") 
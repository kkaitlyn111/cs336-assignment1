import pickle

file_path = "/data/a1-basics/TinyStoriesV2-GPT4-valid.txt"  # Replace with your actual file path

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

words = text.split()
first_1000_words = words[:1000]

print(" ".join(first_1000_words))

# vocab_path = "/data/c-kaitwang/tinystories_vocab.pkl"
# with open(vocab_path, "rb") as f:
#     vocab = pickle.load(f)

# # Print the first 10 tokens and their IDs
# for i, (idx, token) in enumerate(vocab.items()):
#     print(f"Token: {repr(token)}\tID: {idx}")
#     if i == 11:
#         break

# # Build reverse mapping
# id_to_token = {idx: token for idx, token in vocab.items()}
# print(list(id_to_token.keys())[:20])

# # Print the token for ID 10
# if 10 in id_to_token:
#     print(f"Token for ID 10: {repr(id_to_token[10])}")
# else:
#     print("No token found for ID 10")

# import numpy as np

# pretokens_path = "/data/c-kaitwang/tinystories_pretokens.npy"
# tokens = np.load(pretokens_path)
# print(tokens[0:100])  # Print the first 100 tokens
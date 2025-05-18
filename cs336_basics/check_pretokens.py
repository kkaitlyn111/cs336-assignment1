import numpy as np
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check pretokenized .npy file for out-of-bounds token indices.")
    parser.add_argument('--pretokens_path', type=str, required=True, help='Path to pretokenized .npy file')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to vocab .pkl file')
    args = parser.parse_args()

    # load vocab
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    print(f"Loaded vocab size: {vocab_size}")

    # load pretokenized data
    print(f"Loading pretokenized data from {args.pretokens_path} ...")
    tokens = np.load(args.pretokens_path, mmap_mode='r')
    print(f"Pretokenized data shape: {tokens.shape}")

    # check for out-of-bounds indices
    min_token = tokens.min()
    max_token = tokens.max()
    print(f"Token ID range: {min_token} to {max_token}")

    oob_mask = (tokens < 0) | (tokens >= vocab_size)
    num_oob = np.count_nonzero(oob_mask)
    if num_oob == 0:
        print("All token IDs are within the valid range.")
    else:
        print(f"Found {num_oob} out-of-bounds token IDs!")
        # print up to 10 examples
        oob_indices = np.argwhere(oob_mask)
        print("First 10 out-of-bounds indices and their values:")
        for idx in oob_indices[:10]:
            print(f"Index: {tuple(idx)}, Value: {tokens[tuple(idx)]}") 
import os
from typing import BinaryIO
import logging
import time
import numpy as np
import pickle
from cs336_basics.simple_tokenizer import *
from cs336_basics.pretokenization_example import *

def find_chunk_boundaries(
    file: BinaryIO, 
    num_workers: int,
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # Calculate desired number of chunks based on file size
    # Aim for chunks of roughly 1MB each
<<<<<<< HEAD
<<<<<<< HEAD
    #target_chunk_size = 1024 * 1024  # 1MB
    #desired_num_chunks = max(1, min(1000, file_size // target_chunk_size))
    desired_num_chunks = 5000
=======
    # target_chunk_size = 1024 * 1024  # 1MB
    # desired_num_chunks = max(1, min(1000, file_size // target_chunk_size))
    desired_num_chunks = 10000
>>>>>>> ebe672e7df2604172e5fc64531dc1d0a3eeaa5d3
=======
    # target_chunk_size = 1024 * 1024  # 1MB
    # desired_num_chunks = max(1, min(1000, file_size // target_chunk_size))
    desired_num_chunks = 10000
>>>>>>> ebe672e7df2604172e5fc64531dc1d0a3eeaa5d3

    chunk_size = file_size / desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [int(i * chunk_size) for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    print(f"Found {len(chunk_boundaries)-1} chunks")
    return sorted(set(chunk_boundaries))



def pretokenize_file_only(input_path: str, pkl_output_path: str, max_workers: int, special_tokens) -> None:
    """
    Pretokenize a text file and save the pretoken frequency table as a pickle file.
    Args:
        input_path: Path to the input text file
        pkl_output_path: Path to save the pretoken frequency table (.pkl)
        max_workers: Number of workers for parallel pretokenization
        special_tokens: List of special tokens
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Pretokenizing file: {input_path}")
    start_time = time.time()

    # Import here to avoid circular import
    from cs336_basics.train_bpe import read_txt_file

    logger.info("Starting parallel pretokenization...")
    pretoken_freq = read_txt_file(input_path, max_workers, special_tokens)
    logger.info(f"Found {len(pretoken_freq)} unique pretokens")
    total_pretokens = sum(pretoken_freq.values())
    logger.info(f"Processing {total_pretokens} total pretokens")

    # Save pretoken frequency table as .pkl
    with open(pkl_output_path, 'wb') as f:
        pickle.dump(pretoken_freq, f)
    logger.info(f"Saved pretoken frequency table to {pkl_output_path}")
    
    end_time = time.time()
    logger.info(f"Pretokenization completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Saved pretoken freq to {pkl_output_path}")

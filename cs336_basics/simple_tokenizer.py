import os
import pickle
import logging
import time
import multiprocessing
from typing import List, Dict, Tuple, Iterable, Iterator
from tqdm import tqdm
from cs336_basics.train_bpe import train_bpe
import psutil
import regex as re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None, max_workers: int = 4):
        self.vocab = vocab
        self.special_tokens = special_tokens or []
        if special_tokens:
            existing_special_tokens = {}
            for special in special_tokens:
                encoded = special.encode("UTF-8")
                for token_id, token_bytes in self.vocab.items():
                    if token_bytes == encoded:
                        existing_special_tokens[special] = token_id
                        break
            
            new_special_tokens = [st for st in special_tokens if st not in existing_special_tokens]
            if new_special_tokens:
                # start new special tokens at 50256 to match tiktoken
                self.vocab.update({50256 + i: special.encode("UTF-8") for i, special in enumerate(new_special_tokens)})
            
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)

        self.merges = merges
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        self.logger = logging.getLogger(__name__)
        self.max_workers = min(max_workers, multiprocessing.cpu_count())
        self.merges_dict = {pair: pair[0] + pair[1] for pair in merges}
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

        if self.special_tokens:
            # sort special in desc order to ensure longest matches first
            tok_pat = "|".join(re.escape(st) for st in sorted(self.special_tokens, key=len, reverse=True))
            tok_pat = f"({tok_pat})|"
        else:
            tok_pat = ""
        bpe_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._token_re = re.compile(tok_pat + bpe_pat)

        self.logger.info(f"Initialized Tokenizer with max_workers={self.max_workers}")
        self.logger.info(f"Loaded vocabulary size: {len(self.vocab)}")
        self.logger.info(f"Loaded special tokens count: {len(self.special_tokens)}")
    
    @staticmethod
    def get_pairs(word: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
        pairs = set()
        for i in range(len(word)-1):
            pairs.add((word[i], word[i+1]))
        return pairs


    def apply_bpe(self, byte_seq: List[bytes]) -> List[bytes]:
        word = tuple(byte_seq)
        pairs = Tokenizer.get_pairs(word)
        if not pairs:
            return list(word)

        while True:
            # pick the highest‑priority merge that exists in this word
            min_pair = None
            min_rank = None
            for pair in pairs:
                rank = self.bpe_ranks.get(pair)
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_pair, min_rank = pair, rank
            if min_pair is None:
                break

            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                # if first+second occurs here, merge it
                if i < len(word)-1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            pairs = Tokenizer.get_pairs(word)

        return list(word)
    
    def encode(self, text: str) -> List[int]:
        ids = []
        if not self.special_tokens:
            # if no special tokens, just process normally
            for m in self._token_re.finditer(text):
                tok = m.group(0)
                b_list = [bytes([b]) for b in tok.encode("utf-8")]
                for piece in self.apply_bpe(b_list):
                    ids.append(self.vocab_inverse[piece])
            return ids

        # split text by special tokens
        parts = []
        current_pos = 0
        while current_pos < len(text):
            # find next special token
            next_special = None
            next_pos = len(text)
            for special in self.special_tokens:
                pos = text.find(special, current_pos)
                if pos != -1 and pos < next_pos:
                    next_special = special
                    next_pos = pos

            if next_special is not None:
                if next_pos > current_pos:
                    parts.append((text[current_pos:next_pos], False))
                parts.append((next_special, True))
                current_pos = next_pos + len(next_special)
            else:
                if current_pos < len(text):
                    parts.append((text[current_pos:], False))
                break

        for part, is_special in parts:
            if is_special:
                ids.append(self.vocab_inverse[part.encode("utf-8")])
            else:
                for m in self._token_re.finditer(part):
                    tok = m.group(0)
                    b_list = [bytes([b]) for b in tok.encode("utf-8")]
                    for piece in self.apply_bpe(b_list):
                        ids.append(self.vocab_inverse[piece])
        
        return ids
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        """constructs and returns tokenizer from serialized vocabulary and merges
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Loading tokenizer from files: {vocab_filepath} and {merges_filepath}")
        
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
            
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
            
        tokenizer = cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
        logger.info(f"Successfully loaded tokenizer with vocabulary size: {len(vocab)}")
        return tokenizer
    
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """convert an iterable of strings to an iterable of token IDs"""
        self.logger.info("Starting streaming encoding")
        total_tokens = 0
        start_time = time.time()
        
        for string in tqdm(iterable, desc="Streaming encoding"):
            tokens = self.encode(string)
            total_tokens += len(tokens)
            yield from tokens
            
        encoding_time = time.time() - start_time
        self.logger.info(f"Streaming encoding completed in {encoding_time:.2f} seconds")
        self.logger.info(f"Processed {total_tokens} total tokens")
        self.logger.info(f"Average tokens per second: {total_tokens/encoding_time:.2f}")
    
    def decode(self, token_ids: List[int]) -> str:
        """convert token IDs back to text"""
        self.logger.debug(f"Decoding {len(token_ids)} tokens")
        
        tokens = [self.vocab[token_id] for token_id in tqdm(token_ids, desc="Converting IDs to tokens")]
        
        decoded_text = b''.join(tokens).decode('utf-8', errors='replace')
        self.logger.debug(f"Decoded text length: {len(decoded_text)}")
        return decoded_text
    
    def save(self, vocab_path: str, merges_path: str):
        """save the tokenizer's vocab and merges to disk"""
        self.logger.info(f"Saving vocabulary to {vocab_path}")
        start_time = time.time()
        
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.vocab, f)
        
        self.logger.info(f"Saving merges to {merges_path}")
        with open(merges_path, 'wb') as f:
            pickle.dump(self.merges, f)
        
        save_time = time.time() - start_time
        self.logger.info(f"Save completed in {save_time:.2f} seconds")
    
    def load(self, vocab_path: str, merges_path: str):
        """load the tokenizer's vocabulary and merges from disk"""
        self.logger.info(f"Loading vocabulary from {vocab_path}")
        start_time = time.time()
        
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        self.logger.info(f"Loading merges from {merges_path}")
        with open(merges_path, 'rb') as f:
            self.merges = pickle.load(f)
        
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        
        load_time = time.time() - start_time
        self.logger.info(f"Load completed in {load_time:.2f} seconds")
        self.logger.info(f"Loaded vocabulary size: {len(self.vocab)}")
        self.logger.info(f"Loaded merges count: {len(self.merges)}")
    

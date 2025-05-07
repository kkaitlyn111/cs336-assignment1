import pickle
import logging
import os
from cs336_basics.simple_tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_pickle_files():
    # Check if we're using the correct dataset paths
    train_path = "/Users/kaitlynwang/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    valid_path = "/Users/kaitlynwang/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    
    logger.info("Checking dataset paths...")
    if os.path.exists(train_path):
        logger.info(f"Found large dataset train file at: {train_path}")
    else:
        logger.warning(f"Large dataset train file not found at: {train_path}")
    
    if os.path.exists(valid_path):
        logger.info(f"Found large dataset validation file at: {valid_path}")
    else:
        logger.warning(f"Large dataset validation file not found at: {valid_path}")

    # Check vocab file
    logger.info("\nChecking tinystories_vocab.pkl...")
    try:
        with open("tinystories_vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
            logger.info(f"Vocabulary size: {len(vocab)}")
            logger.info("Sample of vocabulary items:")
            for i, (token_id, token_bytes) in enumerate(vocab.items()):
                if i < 5:  # Show first 5 items
                    logger.info(f"Token ID {token_id}: {token_bytes}")
    except Exception as e:
        logger.error(f"Error reading vocab file: {e}")

    # Check merges file
    logger.info("\nChecking tinystories_merges.pkl...")
    try:
        with open("tinystories_merges.pkl", 'rb') as f:
            merges = pickle.load(f)
            logger.info(f"Number of merges: {len(merges)}")
            logger.info("Sample of merges:")
            for i, merge in enumerate(merges):
                if i < 5:  # Show first 5 merges
                    logger.info(f"Merge {i}: {merge}")
    except Exception as e:
        logger.error(f"Error reading merges file: {e}")

    # Test tokenizer with a sample from the large dataset
    logger.info("\nTesting tokenizer with a sample from the large dataset...")
    try:
        tokenizer = Tokenizer.from_files("tinystories_vocab.pkl", "tinystories_merges.pkl")
        
        # Read a small sample from the training file
        with open(train_path, 'r', encoding='utf-8') as f:
            sample_text = f.read(1000)  # Read first 1000 characters
            
        # Tokenize the sample
        token_ids = tokenizer.encode(sample_text)
        logger.info(f"Successfully tokenized sample text into {len(token_ids)} tokens")
        logger.info("Sample of tokenized text (first 10 tokens):")
        for i, token_id in enumerate(token_ids[:10]):
            token = tokenizer.vocab[token_id]
            logger.info(f"Token {i}: ID={token_id}, Bytes={token}")
            
        # Decode back to text
        decoded_text = tokenizer.decode(token_ids[:50])  # Decode first 50 tokens
        logger.info("\nDecoded text sample:")
        logger.info(decoded_text)
        
    except Exception as e:
        logger.error(f"Error testing tokenizer: {e}")

if __name__ == "__main__":
    check_pickle_files() 
# This files defines our utility functions that we used in our notebooks

import os 
import pickle
import pandas as pd
import numpy as np

rule_col = "rule"
body_col = "body"  
label_col = "rule_violation"

def embed_batch(texts, encoder):
    return encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

def load_or_create_embeddings(df, prefix, encoder_name, encoder):
    cache_dir = f"embeddings_cache_{encoder_name}"
    os.makedirs(cache_dir, exist_ok=True)

    embeddings = {}

    to_embed = {
        "body": body_col,
        "pos1": "positive_example_1",
        "pos2": "positive_example_2",
        "neg1": "negative_example_1",
        "neg2": "negative_example_2"
    }

    for key, col in to_embed.items():
        cache_file = f"{cache_dir}/{prefix}_{key}_emb.pkl"

        if os.path.exists(cache_file):
            print(f"Loading cached {prefix} {key}...")
            embeddings[f"{key}_emb"] = pickle.load(open(cache_file, "rb"))

        else:
            print(f"Embedding {prefix} {col}...")
            emb = embed_batch(df[col].tolist(), encoder)
            embeddings[f"{key}_emb"] = emb
            pickle.dump(emb, open(cache_file, "wb"))

    return embeddings

def extract_text_features(text):
    if pd.isna(text):
        text = ""
    
    text_str = str(text)
    
    # Basic length features
    char_count = len(text_str)
    word_count = len(text_str.split())
    
    # Character type counts
    upper_count = sum(1 for c in text_str if c.isupper())
    digit_count = sum(1 for c in text_str if c.isdigit())
    
    # Special character counts
    exclamation_count = text_str.count('!')
    question_count = text_str.count('?')
    
    # URL detection
    has_url = 1 if ('http://' in text_str or 'https://' in text_str or 'www.' in text_str) else 0
    
    # Avg word length
    avg_word_len = char_count / max(word_count, 1)
    
    # Upper case ratio
    upper_ratio = upper_count / max(char_count, 1)
    
    return np.array([
        char_count,
        word_count,
        upper_count,
        digit_count,
        exclamation_count,
        question_count,
        has_url,
        avg_word_len,
        upper_ratio
    ], dtype=np.float32)
    
def combine_features(body, rule, pos1, pos2, neg1, neg2, text, subreddit_encoded):
    sim_pos1 = (body * pos1).sum()
    sim_pos2 = (body * pos2).sum()
    sim_neg1 = (body * neg1).sum()
    sim_neg2 = (body * neg2).sum()
    sim_rule = (body * rule).sum()

    pos_avg = (sim_pos1 + sim_pos2) / 2
    neg_avg = (sim_neg1 + sim_neg2) / 2
    pos_neg_ratio = pos_avg / (neg_avg + 1e-6)
    pos_neg_diff = pos_avg - neg_avg

    text_feats = extract_text_features(text)

    return np.concatenate([
        [sim_pos1, sim_pos2, sim_neg1, sim_neg2, sim_rule], 
        [pos_neg_ratio, pos_neg_diff],  
        text_feats,  
        [subreddit_encoded]  
    ])
# retrieve.py

#Libraries 
import os, json, numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from rank_bm25 import BM25Okapi


#Env Variables 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load chunks 
def load_chunks(path="data/chunks.jsonl"):
    chunks = [json.loads(line) for line in open(path, "r")]
    return chunks

#  Min-Max Normalization for a unified, normalized scoring system for sparse and dense retrieval 
def min_max_norm(scores):
    scores = np.array(scores)
    if np.max(scores) == np.min(scores):
        return np.zeros_like(scores)
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

# Hybrid Retrieval 
def hybrid_retrieve(query, chunks, top_k=10, alpha=0.6):
    tokenized_corpus = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())
    bm25_norm = min_max_norm(bm25_scores)
    bm25_dict = {chunks[i]["id"]: bm25_norm[i] for i in range(len(chunks))}

    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    vec_results = index.query(vector=q_emb, top_k=top_k * 2, include_metadata=True)
    vec_norm_scores = min_max_norm([m.score for m in vec_results.matches])
    vec_dict = {m.id: vec_norm_scores[i] for i, m in enumerate(vec_results.matches)}

    fusion_scores = {
        cid: (1 - alpha) * bm25_dict.get(cid, 0.0) + alpha * vec_dict.get(cid, 0.0)
        for cid in bm25_dict.keys()
    }

    ranked = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [r[0] for r in ranked[:top_k]]
    retrieved = [c for c in chunks if c["id"] in top_ids]

    #print("\n Retrieved Context:")
    #for c in retrieved:
        #print(f" - {c['doc']} (p.{c['page']})")
    return retrieved

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


def get_query_embedding(query: str):
    return model.encode(query + " [SEP] ", normalize_embeddings=True).tolist()


print("\n--- 3. Чистий семантичний пошук ---")
query_text = "teaching machines to recognize objects in pictures"
query_emb = get_query_embedding(query_text)

res = index.query(vector=query_emb, top_k=TOP_K, include_metadata=True)
for match in res["matches"]:
    m = match["metadata"]
    print(f"[{m['year']}] {m['category']} | {m['title']} (Score: {match['score']:.4f})")

print("\n--- 4. Пошук з фільтрацією (Приклад A) ---")
query_a = "reinforcement learning"
query_emb_a = get_query_embedding(query_a)
res_a = index.query(
    vector=query_emb_a,
    top_k=TOP_K,
    include_metadata=True,
    filter={"category": {"$eq": "cs.LG"}, "year": {"$gte": 2019}},
)
for match in res_a["matches"]:
    print(
        f"[{match['metadata']['year']}] {match['metadata']['category']} | {match['metadata']['title']}"
    )

print("\n--- 4. Пошук з фільтрацією (Приклад B) ---")
res_b = index.query(
    vector=query_emb_a,
    top_k=TOP_K,
    include_metadata=True,
    filter={"year": {"$lt": 2015}},
)
for match in res_b["matches"]:
    print(
        f"[{match['metadata']['year']}] {match['metadata']['category']} | {match['metadata']['title']}"
    )

print("\n--- 5. Порівняння локальних метрик ---")
local_embeddings = np.load("embeddings/embeddings.npy")
query_vec_np = np.array(query_emb)

dot_scores = np.dot(local_embeddings, query_vec_np)
cosine_scores = dot_scores
l2_distances = np.linalg.norm(local_embeddings - query_vec_np, axis=1)

top_dot_idx = np.argsort(dot_scores)[::-1][:TOP_K]
top_l2_idx = np.argsort(l2_distances)[:TOP_K]

print("Топ-5 за Dot Product / Cosine Similarity:")
for idx in top_dot_idx:
    print(f"- {df.iloc[idx]['title']} (Score: {dot_scores[idx]:.4f})")

print("\nТоп-5 за L2 Distance:")
for idx in top_l2_idx:
    print(f"- {df.iloc[idx]['title']} (Dist: {l2_distances[idx]:.4f})")

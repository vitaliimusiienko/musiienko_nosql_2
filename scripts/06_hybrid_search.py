import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

print("Підготовка локального BM25 індексу...")
corpus = (df["title"] + " " + df["abstract"]).fillna("").tolist()
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)


def search_bm25(query, k=TOP_K):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    return [
        {
            "id": df.iloc[i]["id"],
            "title": df.iloc[i]["title"],
            "score": scores[i],
            "rank": rank + 1,
        }
        for rank, i in enumerate(top_indices)
    ]


def search_vector(query, k=TOP_K):
    query_emb = model.encode(query + " [SEP] ", normalize_embeddings=True).tolist()
    res = index.query(vector=query_emb, top_k=k, include_metadata=True)
    return [
        {
            "id": m["metadata"]["arxiv_id"],
            "title": m["metadata"]["title"],
            "score": m["score"],
            "rank": rank + 1,
        }
        for rank, m in enumerate(res["matches"])
    ]


def hybrid_rrf_search(query, k_rrf=60, top_k=5):
    bm25_res = search_bm25(query, k=20)
    vec_res = search_vector(query, k=20)

    rrf_scores = {}

    for item in bm25_res:
        doc_id = item["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k_rrf + item["rank"])

    for item in vec_res:
        doc_id = item["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k_rrf + item["rank"])

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    result = []
    for doc_id, score in sorted_docs:
        title = df[df["id"] == doc_id]["title"].values[0]
        result.append({"id": doc_id, "title": title, "rrf_score": score})
    return result


queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]

for q in queries:
    print(f"\n=== ЗАПИТ: '{q}' ===")
    print("\nBM25 Топ-5:")
    for res in search_bm25(q, 5):
        print(f"- {res['title']} (Score: {res['score']:.2f})")

    print("\nВекторний Топ-5:")
    for res in search_vector(q, 5):
        print(f"- {res['title']} (Score: {res['score']:.2f})")

    print("\nГібридний (RRF) Топ-5:")
    for res in hybrid_rrf_search(q, top_k=5):
        print(f"- {res['title']} (RRF: {res['rrf_score']:.4f})")

import os
import re

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

df["abstract_len"] = df["abstract"].apply(lambda x: len(str(x).split()))
longest_papers = df.nlargest(30, "abstract_len").copy()


def fixed_size_chunking(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def semantic_chunking(text, max_words=50):
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks, current_chunk = [], []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_len + sentence_len > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = sentence_len
        else:
            current_chunk.append(sentence)
            current_len += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


indexes = ["arxiv-chunks-fixed", "arxiv-chunks-semantic"]
for idx_name in indexes:
    if idx_name not in pc.list_indexes().names():
        pc.create_index(
            name=idx_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

idx_fixed = pc.Index("arxiv-chunks-fixed")
idx_semantic = pc.Index("arxiv-chunks-semantic")

print("Обробка та завантаження чанків...")
fixed_vectors, semantic_vectors = [], []

for _, row in tqdm(longest_papers.iterrows(), total=len(longest_papers)):
    text = row["abstract"]
    fixed = fixed_size_chunking(text)
    semantic = semantic_chunking(text)

    for i, c_text in enumerate(fixed):
        emb = model.encode(c_text, normalize_embeddings=True).tolist()
        fixed_vectors.append(
            (
                f"{row['id']}_f_{i}",
                emb,
                {
                    "arxiv_id": row["id"],
                    "title": row["title"],
                    "text": c_text[:500],
                    "chunk_id": i,
                },
            )
        )

    for i, c_text in enumerate(semantic):
        emb = model.encode(c_text, normalize_embeddings=True).tolist()
        semantic_vectors.append(
            (
                f"{row['id']}_s_{i}",
                emb,
                {
                    "arxiv_id": row["id"],
                    "title": row["title"],
                    "text": c_text[:500],
                    "chunk_id": i,
                },
            )
        )

for i in range(0, len(fixed_vectors), 100):
    idx_fixed.upsert(vectors=fixed_vectors[i : i + 100])
for i in range(0, len(semantic_vectors), 100):
    idx_semantic.upsert(vectors=semantic_vectors[i : i + 100])

print("\nПошук по чанках...")
q_emb = model.encode(
    "deep neural networks optimization", normalize_embeddings=True
).tolist()

print("\n--- Fixed Chunking ---")
for m in idx_fixed.query(vector=q_emb, top_k=3, include_metadata=True)["matches"]:
    print(f"{m['metadata']['title']} -> {m['metadata']['text'][:100]}...")

print("\n--- Semantic Chunking ---")
for m in idx_semantic.query(vector=q_emb, top_k=3, include_metadata=True)["matches"]:
    print(f"{m['metadata']['title']} -> {m['metadata']['text'][:100]}...")

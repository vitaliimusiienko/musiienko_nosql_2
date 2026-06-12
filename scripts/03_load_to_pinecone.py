import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Створення індексу '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

print("Завантаження локальних даних...")
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

print("Початок завантаження векторів у Pinecone...")
vectors_to_upsert = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    vector_id = f"paper_{row['id']}"

    metadata = {
        "arxiv_id": str(row["id"]),
        "title": str(row["title"]),
        "abstract": str(row["abstract"])[:500],
        "authors": str(row["authors"])[:200],
        "year": int(row["year"]),
        "category": str(row["category"]),
    }

    vectors_to_upsert.append(
        {"id": vector_id, "values": embeddings[i].tolist(), "metadata": metadata}
    )

    if len(vectors_to_upsert) >= BATCH_SIZE:
        index.upsert(vectors=vectors_to_upsert)
        vectors_to_upsert = []

if vectors_to_upsert:
    index.upsert(vectors=vectors_to_upsert)

print("\nЗавантаження завершено!")
stats = index.describe_index_stats()
print(f"Загальна кількість векторів в індексі: {stats['total_vector_count']}")

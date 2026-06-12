import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PARQUET = "data/arxiv_subset.parquet"
OUTPUT_EMBEDDINGS = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

print("Завантаження даних...")
df = pd.read_parquet(INPUT_PARQUET)

print("Підготовка текстів...")
texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

print(f"Завантаження моделі {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print("Генерація ембеддингів...")
embeddings = model.encode(
    texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
)

print(f"\nЗагальна кількість оброблених текстів: {len(texts)}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.4f}")

os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS), exist_ok=True)
np.save(OUTPUT_EMBEDDINGS, embeddings)
print(f"Ембеддинги успішно збережено у {OUTPUT_EMBEDDINGS}")

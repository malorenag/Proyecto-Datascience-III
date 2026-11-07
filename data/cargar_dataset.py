"""
Loader de los datasets de la cátedra (Data Science III: NLP & Deep Learning).
Sirve para las pre-entregas de los Módulos 2, 3, 4 y el Proyecto Final.

Elegí UN dataset al inicio del Módulo 2 y usá el MISMO hasta el final.
Los splits train/test ya vienen separados: NUNCA entrenes con el test.
"""

import pandas as pd
from pathlib import Path

# Cambiá a "bbc_news" si preferís artículos largos.
DATASET = "ag_news"            # opciones: "ag_news" | "bbc_news"
BASE = Path(__file__).parent   # carpeta donde está este script


def cargar(dataset: str = DATASET):
    """Devuelve (train_df, test_df) con columnas: text, label."""
    tr = pd.read_csv(BASE / dataset / f"{dataset}_train.csv")
    te = pd.read_csv(BASE / dataset / f"{dataset}_test.csv")
    return tr, te


if __name__ == "__main__":
    train_df, test_df = cargar()
    print(f"Dataset: {DATASET}")
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")
    print("Clases:", sorted(train_df['label'].unique()))
    print(train_df.head())

# ---------------------------------------------------------------------------
# Para el Módulo 4 (fine-tuning con Hugging Face + LoRA), convertí a Dataset HF:
#
#   from datasets import Dataset, DatasetDict
#   train_df, test_df = cargar("ag_news")
#   # el Trainer necesita labels enteros:
#   clases = sorted(train_df["label"].unique())
#   c2i = {c: i for i, c in enumerate(clases)}
#   train_df["labels"] = train_df["label"].map(c2i)
#   test_df["labels"]  = test_df["label"].map(c2i)
#   ds = DatasetDict({
#       "train": Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False),
#       "test":  Dataset.from_pandas(test_df[["text", "labels"]],  preserve_index=False),
#   })
# ---------------------------------------------------------------------------


import os
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

raw_path = os.getenv("DATASET_PATH")

if not raw_path:
    raise ValueError("Configure DATASET_PATH no arquivo .env")

dataset_path = Path(raw_path)

if not dataset_path.exists():
    raise FileNotFoundError(dataset_path)

extensoes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

contagem = {}

for pasta in dataset_path.iterdir():

    if pasta.is_dir():

        imagens = [
            arquivo for arquivo in pasta.rglob("*")
            if arquivo.is_file()
            and arquivo.suffix.lower() in extensoes
        ]

        contagem[pasta.name] = len(imagens)

print("\nTOTAL DE CATEGORIAS:", len(contagem))
print("TOTAL DE IMAGENS:", sum(contagem.values()))

print("\nCATEGORIAS COM MENOS IMAGENS:")

for nome, quantidade in sorted(
    contagem.items(),
    key=lambda item: item[1]
)[:30]:

    print(f"{nome}: {quantidade} imagens")

print("\nCATEGORIAS SEM IMAGENS:",
      sum(1 for n in contagem.values() if n == 0))

print("\nCATEGORIAS COM UMA IMAGEM:",
      sum(1 for n in contagem.values() if n == 1))

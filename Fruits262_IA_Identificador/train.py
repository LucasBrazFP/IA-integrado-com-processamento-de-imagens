
import os
import csv
import random
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from dotenv import load_dotenv

SEED = 42
BATCH_SIZE = 16
EPOCHS = 40
PATIENCE = 8
IMAGE_SIZE = 224
LIMITE_CONFIANCA = 0.70

EPOCAS_BUSCA = 5

CONFIGURACOES = [
    {"lr": 0.0001, "weight_decay": 0.01},
    {"lr": 0.0003, "weight_decay": 0.01},
    {"lr": 0.0001, "weight_decay": 0.001},
]

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")

if not DATASET_PATH or not os.path.isdir(DATASET_PATH):
    raise ValueError(
        "Caminho do dataset inválido. Verifique o arquivo .env."
    )

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo:", DEVICE)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1)
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class DatasetComTransform(Dataset):

    def __init__(self, dataset, transform, retornar_indice=False):
        self.dataset = dataset
        self.transform = transform
        self.retornar_indice = retornar_indice

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        imagem, rotulo = self.dataset[index]

        if self.transform:
            imagem = self.transform(imagem)

        if self.retornar_indice:
            return imagem, rotulo, index

        return imagem, rotulo

def dividir_dataset(dataset):

    indices_por_classe = defaultdict(list)

    for indice, (_, rotulo) in enumerate(dataset.samples):
        indices_por_classe[rotulo].append(indice)

    indices_treino = []
    indices_validacao = []

    gerador = random.Random(SEED)

    for rotulo, indices in indices_por_classe.items():

        gerador.shuffle(indices)

        if len(indices) < 2:
            indices_treino.extend(indices)
            continue

        quantidade_validacao = max(
            1,
            min(
                len(indices) - 1,
                round(len(indices) * 0.2)
            )
        )

        indices_validacao.extend(
            indices[:quantidade_validacao]
        )

        indices_treino.extend(
            indices[quantidade_validacao:]
        )

    gerador.shuffle(indices_treino)
    gerador.shuffle(indices_validacao)

    return indices_treino, indices_validacao


def criar_modelo(quantidade_classes):

    modelo = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )

    modelo.classifier[3] = nn.Linear(
        modelo.classifier[3].in_features,
        quantidade_classes
    )

    return modelo.to(DEVICE)
def avaliar(modelo, carregador, criterio):

    modelo.eval()

    perda_total = 0
    acertos = 0
    total = 0

    with torch.no_grad():

        for imagens, rotulos in carregador:

            imagens = imagens.to(DEVICE)
            rotulos = rotulos.to(DEVICE)

            saidas = modelo(imagens)

            perda = criterio(saidas, rotulos)

            perda_total += perda.item() * imagens.size(0)

            previsoes = saidas.argmax(dim=1)

            acertos += (
                previsoes == rotulos
            ).sum().item()

            total += rotulos.size(0)

    return perda_total / total, 100 * acertos / total
def buscar_hiperparametros(
    quantidade_classes,
    carregador_treino,
    carregador_validacao
):

    print("\nIniciando busca de hiperparâmetros...")

    melhor_perda = float("inf")
    melhor_configuracao = CONFIGURACOES[0]

    for numero, configuracao in enumerate(
        CONFIGURACOES, start=1
    ):

        print(
            f"\nTeste {numero}/{len(CONFIGURACOES)}"
        )

        print("Configuração:", configuracao)

        modelo = criar_modelo(quantidade_classes)

        criterio = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

        otimizador = torch.optim.AdamW(
            modelo.parameters(),
            lr=configuracao["lr"],
            weight_decay=configuracao["weight_decay"]
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            otimizador,
            mode="min",
            factor=0.5,
            patience=2
        )

        melhor_perda_teste = float("inf")

        for epoca in range(EPOCAS_BUSCA):

            modelo.train()

            for imagens, rotulos, _ in carregador_treino:

                imagens = imagens.to(DEVICE)
                rotulos = rotulos.to(DEVICE)

                otimizador.zero_grad()

                saidas = modelo(imagens)

                perda = criterio(saidas, rotulos)

                perda.backward()

                otimizador.step()

            perda_validacao, acuracia = avaliar(
                modelo,
                carregador_validacao,
                criterio
            )

            scheduler.step(perda_validacao)

            melhor_perda_teste = min(
                melhor_perda_teste,
                perda_validacao
            )

            print(
                f"Época {epoca + 1}/{EPOCAS_BUSCA} | "
                f"Perda: {perda_validacao:.4f} | "
                f"Acurácia: {acuracia:.2f}%"
            )

        if melhor_perda_teste < melhor_perda:

            melhor_perda = melhor_perda_teste

            melhor_configuracao = configuracao.copy()

    print("\nMelhor configuração encontrada:")
    print(melhor_configuracao)

    return melhor_configuracao

def criar_carregador_com_pesos(
    dataset_treino,
    pesos_amostras
):

    pesos = pesos_amostras.clone()

    pesos = pesos / pesos.mean()

    pesos = torch.clamp(
        pesos,
        min=0.5,
        max=3.0
    )

    sampler = WeightedRandomSampler(
        weights=pesos,
        num_samples=len(dataset_treino),
        replacement=True
    )

    return DataLoader(
        dataset_treino,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0
    )


def treinar_epoca(
    modelo,
    carregador,
    criterio,
    otimizador,
    pesos_amostras
):

    modelo.train()

    perda_total = 0
    acertos = 0
    total = 0

    for imagens, rotulos, indices in carregador:

        imagens = imagens.to(DEVICE)
        rotulos = rotulos.to(DEVICE)

        otimizador.zero_grad()

        saidas = modelo(imagens)

        perdas = criterio(
            saidas,
            rotulos
        )
        perda = perdas.mean()

        perda.backward()

        otimizador.step()

        with torch.no_grad():

            indices = indices.long()

            perdas_cpu = perdas.detach().cpu()

            perdas_antigas = pesos_amostras[indices]

            pesos_amostras[indices] = (
                0.7 * perdas_antigas
                + 0.3 * perdas_cpu
            )

        perda_total += perda.item() * imagens.size(0)

        previsoes = saidas.argmax(dim=1)

        acertos += (
            previsoes == rotulos
        ).sum().item()

        total += rotulos.size(0)

    return (
        perda_total / total,
        100 * acertos / total
    )

def gerar_relatorios(modelo, carregador, classes):

    modelo.eval()

    matriz = torch.zeros(
        len(classes),
        len(classes),
        dtype=torch.int64
    )

    previsoes_detalhadas = []

    with torch.no_grad():

        for imagens, rotulos in carregador:

            imagens = imagens.to(DEVICE)

            saidas = modelo(imagens)

            probabilidades = torch.softmax(
                saidas,
                dim=1
            )

            confiancas, previsoes = probabilidades.max(
                dim=1
            )

            previsoes = previsoes.cpu()
            confiancas = confiancas.cpu()

            for real, previsto, confianca in zip(
                rotulos,
                previsoes,
                confiancas
            ):

                matriz[real.item(), previsto.item()] += 1

                status = (
                    "CONFIÁVEL"
                    if confianca.item() >= LIMITE_CONFIANCA
                    else "BAIXA CONFIANÇA"
                )

                previsoes_detalhadas.append([
                    classes[real.item()],
                    classes[previsto.item()],
                    round(confianca.item() * 100, 2),
                    status
                ])

    os.makedirs("resultados", exist_ok=True)

    with open(
        "resultados/relatorio_classes_enhanced.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.writer(arquivo)

        escritor.writerow([
            "Fruta",
            "Imagens_validacao",
            "Acertos",
            "Acuracia",
            "Precisao",
            "Recall",
            "F1_score"
        ])

        for indice, classe in enumerate(classes):

            verdadeiros_positivos = matriz[
                indice, indice
            ].item()

            total_real = matriz[indice].sum().item()

            total_previsto = matriz[:, indice].sum().item()

            falsos_positivos = (
                total_previsto - verdadeiros_positivos
            )

            falsos_negativos = (
                total_real - verdadeiros_positivos
            )

            acuracia = (
                100 * verdadeiros_positivos / total_real
                if total_real > 0 else 0
            )

            precisao = (
                verdadeiros_positivos /
                total_previsto
                if total_previsto > 0 else 0
            )

            recall = (
                verdadeiros_positivos /
                total_real
                if total_real > 0 else 0
            )

            f1 = (
                2 * precisao * recall /
                (precisao + recall)
                if precisao + recall > 0 else 0
            )

            escritor.writerow([
                classe,
                total_real,
                verdadeiros_positivos,
                round(acuracia, 2),
                round(precisao, 4),
                round(recall, 4),
                round(f1, 4)
            ])

    with open(
        "resultados/matriz_confusao_enhanced.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.writer(arquivo)

        escritor.writerow(
            ["Real/Previsto"] + classes
        )

        for indice, classe in enumerate(classes):

            escritor.writerow(
                [classe] + matriz[indice].tolist()
            )
    with open(
        "resultados/confianca_previsoes.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.writer(arquivo)

        escritor.writerow([
            "Classe_real",
            "Classe_prevista",
            "Confianca_percentual",
            "Decisao"
        ])

        escritor.writerows(previsoes_detalhadas)

    print("\nRelatórios salvos na pasta resultados.")

def main():

    print("\nCarregando dataset...")

    dataset_original = datasets.ImageFolder(
        DATASET_PATH
    )

    classes = dataset_original.classes

    quantidade_classes = len(classes)

    print("Categorias encontradas:", quantidade_classes)
    print("Total de imagens:", len(dataset_original))

    indices_treino, indices_validacao = dividir_dataset(
        dataset_original
    )

    dataset_treino = DatasetComTransform(
        Subset(dataset_original, indices_treino),
        train_transform,
        retornar_indice=True
    )

    dataset_validacao = DatasetComTransform(
        Subset(dataset_original, indices_validacao),
        val_transform
    )

    carregador_busca = DataLoader(
        dataset_treino,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    carregador_validacao = DataLoader(
        dataset_validacao,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("\nImagens de treinamento:", len(dataset_treino))
    print("Imagens de validação:", len(dataset_validacao))

    melhor_configuracao = buscar_hiperparametros(
        quantidade_classes,
        carregador_busca,
        carregador_validacao
    )

    LEARNING_RATE = melhor_configuracao["lr"]

    WEIGHT_DECAY = melhor_configuracao["weight_decay"]

    print("\nCriando modelo final...")

    modelo = criar_modelo(quantidade_classes)

    criterio = nn.CrossEntropyLoss(
        label_smoothing=0.05,
        reduction="none"
    )

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador,
        mode="min",
        factor=0.5,
        patience=3
    )

    pesos_amostras = torch.ones(
        len(dataset_treino),
        dtype=torch.float32
    )

    melhor_perda = float("inf")
    epocas_sem_melhora = 0

    os.makedirs("modelo", exist_ok=True)

    caminho_modelo = (
        "modelo/fruits262_mobilenet_enhanced.pth"
    )

    print("\nIniciando treinamento final...\n")

    for epoca in range(EPOCHS):

        carregador_treino = criar_carregador_com_pesos(
            dataset_treino,
            pesos_amostras
        )

        perda_treino, acuracia_treino = treinar_epoca(
            modelo,
            carregador_treino,
            criterio,
            otimizador,
            pesos_amostras
        )

        perda_validacao, acuracia_validacao = avaliar(
            modelo,
            carregador_validacao,
            nn.CrossEntropyLoss()
        )

        scheduler.step(perda_validacao)

        print(
            f"Época {epoca + 1}/{EPOCHS} | "
            f"Perda treino: {perda_treino:.4f} | "
            f"Acurácia treino: {acuracia_treino:.2f}% | "
            f"Perda validação: {perda_validacao:.4f} | "
            f"Acurácia validação: {acuracia_validacao:.2f}%"
        )

        if perda_validacao < melhor_perda:

            melhor_perda = perda_validacao

            epocas_sem_melhora = 0

            torch.save({
                "model_state": modelo.state_dict(),
                "classes": classes,
                "image_size": IMAGE_SIZE,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "confidence_threshold": LIMITE_CONFIANCA,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY
            }, caminho_modelo)

            print("Melhor modelo salvo!")

        else:

            epocas_sem_melhora += 1

        if epocas_sem_melhora >= PATIENCE:

            print(
                "\nTreinamento interrompido por falta de melhora."
            )

            break


    print("\nCarregando melhor modelo...")

    checkpoint = torch.load(
        caminho_modelo,
        map_location=DEVICE,
        weights_only=False
    )

    modelo.load_state_dict(
        checkpoint["model_state"]
    )
    gerar_relatorios(
        modelo,
        carregador_validacao,
        classes
    )

    print("\nTreinamento finalizado!")
    print("Modelo:", caminho_modelo)
    print("Melhor perda:", melhor_perda)

if __name__ == "__main__":
    main()

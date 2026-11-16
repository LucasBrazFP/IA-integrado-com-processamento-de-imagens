
import csv
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image


# ==========================================
# CONFIGURAÇÕES
# ==========================================

load_dotenv()

raw_path = os.getenv("path_pictures")

if not raw_path:
    raise ValueError(
        "A variável 'path_pictures' não foi encontrada no .env."
    )

PASTA_ORIGINAL = Path(raw_path).expanduser()
PASTA_BASE = PASTA_ORIGINAL.parent

PASTA_ENTRADA = (
    PASTA_BASE / "Processed_Images_Enhanced"
)

PASTA_MASCARAS = (
    PASTA_BASE / "Segmentation_Masks_Enhanced"
)

PASTA_SAIDA = (
    PASTA_BASE / "FFT_Results_Enhanced"
)

ARQUIVO_METADATA = (
    PASTA_BASE / "fft_metadata_enhanced.csv"
)

ARQUIVO_CARACTERISTICAS = (
    PASTA_BASE / "fft_features.csv"
)

EXTENSOES_VALIDAS = {
    ".png",
    ".jpg",
    ".jpeg"
}

# Raio normalizado utilizado para separar
# baixas e altas frequências.
RAIO_BAIXA_FREQUENCIA = 0.15

# Quantidade de regiões da análise radial.
NUM_BINS_RADIAIS = 16

# Limite para considerar pixels da máscara.
LIMIAR_MASCARA = 20


# ==========================================
# LEITURA DA IMAGEM
# ==========================================

def carregar_imagem(caminho):

    imagem = Image.open(
        caminho
    ).convert("L")

    return np.asarray(
        imagem,
        dtype=np.float32
    )


def carregar_mascara(caminho):

    mascara = Image.open(
        caminho
    ).convert("L")

    return np.asarray(
        mascara,
        dtype=np.float32
    )


# ==========================================
# PREPARAÇÃO DA REGIÃO DE INTERESSE
# ==========================================


def preparar_regiao(imagem, mascara):

    # Redimensiona a máscara para coincidir
    # com a imagem processada (300x300).
    if imagem.shape != mascara.shape:

        mascara_pil = Image.fromarray(
            mascara.astype(np.uint8)
        )

        mascara_pil = mascara_pil.resize(
            (imagem.shape[1], imagem.shape[0]),
            Image.Resampling.LANCZOS
        )

        mascara = np.asarray(
            mascara_pil,
            dtype=np.float32
        )

    mascara_binaria = (
        mascara > LIMIAR_MASCARA
    )

    coordenadas = np.argwhere(
        mascara_binaria
    )

    if coordenadas.size == 0:

        raise ValueError(
            "A máscara não contém pixels da fruta."
        )

    y_min, x_min = coordenadas.min(
        axis=0
    )

    y_max, x_max = (
        coordenadas.max(axis=0) + 1
    )

    recorte = imagem[
        y_min:y_max,
        x_min:x_max
    ]

    mascara_recorte = mascara_binaria[
        y_min:y_max,
        x_min:x_max
    ]

    fruta = recorte * mascara_recorte

    return fruta, mascara_recorte


# ==========================================
# CÁLCULO DA FFT
# ==========================================

def calcular_fft(fruta):

    altura, largura = fruta.shape

    # Remove a média para reduzir
    # a influência do brilho global.
    fruta = fruta - np.mean(fruta)

    # Janela Hann para reduzir
    # efeitos artificiais nas bordas.
    janela_y = (
        np.hanning(altura)
        if altura > 1
        else np.ones(altura)
    )

    janela_x = (
        np.hanning(largura)
        if largura > 1
        else np.ones(largura)
    )

    janela = np.outer(
        janela_y,
        janela_x
    )

    fruta_janelada = fruta * janela

    transformada = np.fft.fft2(
        fruta_janelada
    )

    espectro = np.fft.fftshift(
        transformada
    )

    magnitude = np.abs(
        espectro
    )

    magnitude_log = np.log1p(
        magnitude
    )

    return (
        espectro,
        magnitude,
        magnitude_log,
        largura,
        altura
    )


# ==========================================
# NORMALIZAÇÃO VISUAL
# ==========================================

def normalizar_espectro(magnitude_log):

    minimo = magnitude_log.min()
    maximo = magnitude_log.max()

    normalizada = (
        (magnitude_log - minimo)
        / (maximo - minimo + 1e-8)
    )

    return Image.fromarray(
        np.clip(
            normalizada * 255,
            0,
            255
        ).astype(np.uint8)
    )


# ==========================================
# SEPARAÇÃO DE FREQUÊNCIAS
# ==========================================

def separar_frequencias(magnitude):

    altura, largura = magnitude.shape

    centro_y = altura // 2
    centro_x = largura // 2

    y, x = np.ogrid[
        :altura,
        :largura
    ]

    # Distância normalizada em relação
    # ao centro do espectro.
    distancia = np.sqrt(
        ((y - centro_y) / max(centro_y, 1)) ** 2
        +
        ((x - centro_x) / max(centro_x, 1)) ** 2
    )

    mascara_baixa = (
        distancia <= RAIO_BAIXA_FREQUENCIA
    )

    mascara_alta = (
        distancia > RAIO_BAIXA_FREQUENCIA
    )

    energia = magnitude ** 2

    energia_total = np.sum(
        energia
    )

    energia_baixa = np.sum(
        energia[mascara_baixa]
    )

    energia_alta = np.sum(
        energia[mascara_alta]
    )

    proporcao_baixa = (
        energia_baixa / (energia_total + 1e-8)
    )

    proporcao_alta = (
        energia_alta / (energia_total + 1e-8)
    )

    return {
        "energia_total": float(energia_total),
        "energia_baixa": float(energia_baixa),
        "energia_alta": float(energia_alta),
        "proporcao_baixa": float(proporcao_baixa),
        "proporcao_alta": float(proporcao_alta)
    }, mascara_baixa, mascara_alta


# ==========================================
# ANÁLISE RADIAL
# ==========================================

def analisar_radial(magnitude):

    altura, largura = magnitude.shape

    centro_y = altura // 2
    centro_x = largura // 2

    y, x = np.indices(
        magnitude.shape
    )

    distancia = np.sqrt(
        ((y - centro_y) / max(centro_y, 1)) ** 2
        +
        ((x - centro_x) / max(centro_x, 1)) ** 2
    )

    energia = magnitude ** 2

    limites = np.linspace(
        0,
        distancia.max() + 1e-8,
        NUM_BINS_RADIAIS + 1
    )

    caracteristicas = {}

    for indice in range(
        NUM_BINS_RADIAIS
    ):

        regiao = (
            (distancia >= limites[indice])
            &
            (distancia < limites[indice + 1])
        )

        energia_regiao = np.sum(
            energia[regiao]
        )

        energia_total = np.sum(
            energia
        )

        proporcao = (
            energia_regiao
            / (energia_total + 1e-8)
        )

        caracteristicas[
            f"energia_radial_{indice + 1}"
        ] = float(energia_regiao)

        caracteristicas[
            f"proporcao_radial_{indice + 1}"
        ] = float(proporcao)

    return caracteristicas


# ==========================================
# CARACTERÍSTICAS COMPLEMENTARES
# ==========================================

def extrair_caracteristicas(
    magnitude,
    magnitude_log,
    mascara
):

    caracteristicas = {}

    energia = magnitude ** 2

    energia_total = np.sum(
        energia
    )

    # Energia média.
    caracteristicas[
        "energia_media"
    ] = float(
        np.mean(energia)
    )

    # Desvio padrão do espectro.
    caracteristicas[
        "desvio_padrao_espectro"
    ] = float(
        np.std(magnitude_log)
    )

    # Frequência dominante.
    indice_maximo = np.unravel_index(
        np.argmax(magnitude),
        magnitude.shape
    )

    caracteristicas[
        "frequencia_dominante_y"
    ] = int(
        indice_maximo[0]
    )

    caracteristicas[
        "frequencia_dominante_x"
    ] = int(
        indice_maximo[1]
    )

    # Quantidade de pixels da máscara.
    caracteristicas[
        "area_mascara"
    ] = int(
        np.sum(mascara)
    )

    caracteristicas[
        "energia_espectral_total"
    ] = float(
        energia_total
    )

    return caracteristicas


# ==========================================
# SALVAMENTO DOS ESPECTROS
# ==========================================

def salvar_espectros(
    magnitude_log,
    mascara_baixa,
    mascara_alta,
    pasta_saida,
    nome
):

    pasta_saida.mkdir(
        parents=True,
        exist_ok=True
    )

    # Espectro completo.
    espectro = normalizar_espectro(
        magnitude_log
    )

    espectro.save(
        pasta_saida / f"{nome}_FFT.png"
    )

    # Espectro de baixas frequências.
    baixa = magnitude_log.copy()

    baixa[~mascara_baixa] = 0

    imagem_baixa = normalizar_espectro(
        baixa
    )

    imagem_baixa.save(
        pasta_saida / f"{nome}_baixas.png"
    )

    # Espectro de altas frequências.
    alta = magnitude_log.copy()

    alta[~mascara_alta] = 0

    imagem_alta = normalizar_espectro(
        alta
    )

    imagem_alta.save(
        pasta_saida / f"{nome}_altas.png"
    )


# ==========================================
# PROCESSAMENTO DE UMA IMAGEM
# ==========================================

def processar_imagem(
    caminho_imagem,
    caminho_mascara,
    pasta_saida
):

    imagem = carregar_imagem(
        caminho_imagem
    )

    mascara = carregar_mascara(
        caminho_mascara
    )

    fruta, mascara_recorte = preparar_regiao(
        imagem,
        mascara
    )

    (
        espectro,
        magnitude,
        magnitude_log,
        largura,
        altura
    ) = calcular_fft(fruta)

    (
        caracteristicas_frequencia,
        mascara_baixa,
        mascara_alta
    ) = separar_frequencias(
        magnitude
    )

    caracteristicas_radiais = analisar_radial(
        magnitude
    )

    caracteristicas_complementares = (
        extrair_caracteristicas(
            magnitude,
            magnitude_log,
            mascara_recorte
        )
    )

    salvar_espectros(
        magnitude_log,
        mascara_baixa,
        mascara_alta,
        pasta_saida,
        caminho_imagem.stem
    )

    caracteristicas = {}

    caracteristicas.update(
        caracteristicas_frequencia
    )

    caracteristicas.update(
        caracteristicas_radiais
    )

    caracteristicas.update(
        caracteristicas_complementares
    )

    return caracteristicas, largura, altura


# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================

def main():

    if not PASTA_ENTRADA.exists():

        raise FileNotFoundError(
            f"Pasta não encontrada: {PASTA_ENTRADA}. "
            "Execute main.py atualizado primeiro."
        )

    if not PASTA_MASCARAS.exists():

        raise FileNotFoundError(
            f"Pasta de máscaras não encontrada: "
            f"{PASTA_MASCARAS}"
        )

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    linhas_metadata = []
    linhas_caracteristicas = []

    total = 0
    erros = 0

    for pasta_categoria in sorted(
        PASTA_ENTRADA.iterdir()
    ):

        if not pasta_categoria.is_dir():
            continue

        pasta_mascaras_categoria = (
            PASTA_MASCARAS /
            pasta_categoria.name
        )

        pasta_saida_categoria = (
            PASTA_SAIDA /
            pasta_categoria.name
        )

        pasta_saida_categoria.mkdir(
            parents=True,
            exist_ok=True
        )

        for caminho in sorted(
            pasta_categoria.iterdir()
        ):

            if (
                caminho.suffix.lower()
                not in EXTENSOES_VALIDAS
            ):
                continue

            caminho_mascara = (
                pasta_mascaras_categoria /
                caminho.name
            )

            dados_metadata = {
                "categoria": pasta_categoria.name,
                "imagem_processada": str(
                    caminho.relative_to(PASTA_BASE)
                ),
                "mascara": str(
                    caminho_mascara.relative_to(PASTA_BASE)
                ),
                "status": "erro",
                "erro": ""
            }

            try:

                if not caminho_mascara.exists():

                    raise FileNotFoundError(
                        f"Máscara não encontrada: "
                        f"{caminho_mascara}"
                    )

                caracteristicas, largura, altura = (
                    processar_imagem(
                        caminho,
                        caminho_mascara,
                        pasta_saida_categoria
                    )
                )

                linha = {
                    "categoria": pasta_categoria.name,
                    "imagem": caminho.name,
                    "largura_roi": largura,
                    "altura_roi": altura
                }

                linha.update(
                    caracteristicas
                )

                linhas_caracteristicas.append(
                    linha
                )

                dados_metadata.update({
                    "status": "sucesso",
                    "erro": ""
                })

                total += 1

            except Exception as erro:

                erros += 1

                dados_metadata["erro"] = str(
                    erro
                )

                print(
                    f"ERRO em {caminho}: {erro}"
                )

            linhas_metadata.append(
                dados_metadata
            )

        print(
            f"Categoria concluída: "
            f"{pasta_categoria.name}"
        )

    # ======================================
    # CSV DE METADADOS
    # ======================================

    campos_metadata = [
        "categoria",
        "imagem_processada",
        "mascara",
        "status",
        "erro"
    ]

    with ARQUIVO_METADATA.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=campos_metadata
        )

        escritor.writeheader()

        escritor.writerows(
            linhas_metadata
        )

    # ======================================
    # CSV DE CARACTERÍSTICAS
    # ======================================

    if linhas_caracteristicas:

        campos = list(
            linhas_caracteristicas[0].keys()
        )

        with ARQUIVO_CARACTERISTICAS.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as arquivo:

            escritor = csv.DictWriter(
                arquivo,
                fieldnames=campos
            )

            escritor.writeheader()

            escritor.writerows(
                linhas_caracteristicas
            )

    # ======================================
    # RESUMO
    # ======================================

    print(
        "\n========== FFT CONCLUÍDA =========="
    )

    print(
        f"Espectros processados: {total}"
    )

    print(
        f"Erros: {erros}"
    )

    print(
        f"Resultados: {PASTA_SAIDA}"
    )

    print(
        f"Metadados: {ARQUIVO_METADATA}"
    )

    print(
        f"Características: "
        f"{ARQUIVO_CARACTERISTICAS}"
    )


if __name__ == "__main__":
    main()


import csv
import io
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from PIL import (
    Image,
    ImageEnhance,
    ImageFilter
)
from rembg import new_session, remove


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

if not PASTA_ORIGINAL.exists():
    raise FileNotFoundError(
        f"Pasta de imagens não encontrada: {PASTA_ORIGINAL}"
    )

PASTA_BASE = PASTA_ORIGINAL.parent

# Novas pastas para preservar o processamento anterior.
PASTA_PROCESSADAS = (
    PASTA_BASE / "Processed_Images_Enhanced"
)

PASTA_MASCARAS = (
    PASTA_BASE / "Segmentation_Masks_Enhanced"
)

ARQUIVO_METADATA = (
    PASTA_BASE / "metadata_enhanced.csv"
)


TAMANHO_FINAL = (300, 300)

MODELO = "u2netp"

CONTRASTE = 1.10

# Margem ao redor da fruta no enquadramento.
MARGEM_FRUTA = 0.08

# Operações morfológicas.
TAMANHO_MORFOLOGIA = 3

# Suavização leve da máscara.
RAIO_SUAVIZACAO_MASCARA = 0.5

# Redução de ruído.
TAMANHO_FILTRO_MEDIANA = 3

# Realce de detalhes.
RAIO_REALCE = 1.2
INTENSIDADE_REALCE = 110
LIMIAR_REALCE = 3

# Limite de imagens por categoria.
# None = todas as imagens.
LIMITE_POR_CATEGORIA = 300

# Quantidade de tarefas simultâneas.
MAX_WORKERS = 100

EXTENSOES_VALIDAS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ==========================================
# LISTAGEM DE IMAGENS
# ==========================================

def listar_imagens(pasta):

    return sorted(
        arquivo
        for arquivo in pasta.iterdir()
        if (
            arquivo.is_file()
            and arquivo.suffix.lower()
            in EXTENSOES_VALIDAS
        )
    )


# ==========================================
# SEGMENTAÇÃO
# ==========================================

def segmentar_imagem(caminho, sessao):

    with caminho.open("rb") as arquivo:
        conteudo = arquivo.read()

    resultado = remove(
        conteudo,
        session=sessao
    )

    return Image.open(
        io.BytesIO(resultado)
    ).convert("RGBA")


# ==========================================
# MELHORIA DA MÁSCARA
# ==========================================

def melhorar_mascara(imagem_rgba):

    mascara = imagem_rgba.getchannel("A")

    # Fechamento morfológico:
    # dilatação seguida de erosão.
    mascara = mascara.filter(
        ImageFilter.MaxFilter(
            TAMANHO_MORFOLOGIA
        )
    )

    mascara = mascara.filter(
        ImageFilter.MinFilter(
            TAMANHO_MORFOLOGIA
        )
    )

    # Abertura morfológica:
    # erosão seguida de dilatação.
    mascara = mascara.filter(
        ImageFilter.MinFilter(
            TAMANHO_MORFOLOGIA
        )
    )

    mascara = mascara.filter(
        ImageFilter.MaxFilter(
            TAMANHO_MORFOLOGIA
        )
    )

    # Suavização leve das bordas.
    mascara = mascara.filter(
        ImageFilter.GaussianBlur(
            radius=RAIO_SUAVIZACAO_MASCARA
        )
    )

    imagem_rgba.putalpha(mascara)

    return imagem_rgba


# ==========================================
# ENQUADRAMENTO DA FRUTA
# ==========================================

def obter_caixa_fruta(mascara):

    # Considera pixels com alfa acima de 20.
    mascara_binaria = mascara.point(
        lambda pixel: 255 if pixel > 20 else 0
    )

    caixa = mascara_binaria.getbbox()

    if caixa is None:
        raise ValueError(
            "A máscara não identificou a fruta."
        )

    return caixa


def enquadrar_fruta(imagem_rgba):

    mascara = imagem_rgba.getchannel("A")

    x_min, y_min, x_max, y_max = (
        obter_caixa_fruta(mascara)
    )

    largura = x_max - x_min
    altura = y_max - y_min

    # Calcula margem proporcional.
    margem_x = int(largura * MARGEM_FRUTA)
    margem_y = int(altura * MARGEM_FRUTA)

    x_min = max(0, x_min - margem_x)
    y_min = max(0, y_min - margem_y)

    x_max = min(
        imagem_rgba.width,
        x_max + margem_x
    )

    y_max = min(
        imagem_rgba.height,
        y_max + margem_y
    )

    # Recorta somente a região ocupada pela fruta.
    imagem_recortada = imagem_rgba.crop(
        (x_min, y_min, x_max, y_max)
    )

    return imagem_recortada


# ==========================================
# REDUÇÃO DE RUÍDOS E REALCE
# ==========================================

def melhorar_detalhes(imagem):

    # Redução leve de ruídos.
    imagem = imagem.filter(
        ImageFilter.MedianFilter(
            size=TAMANHO_FILTRO_MEDIANA
        )
    )

    # Realce de bordas e detalhes.
    imagem = imagem.filter(
        ImageFilter.UnsharpMask(
            radius=RAIO_REALCE,
            percent=INTENSIDADE_REALCE,
            threshold=LIMIAR_REALCE
        )
    )

    return imagem


# ==========================================
# PREPARAÇÃO DA IMAGEM
# ==========================================

def preparar_imagem(imagem_rgba):

    # Melhora a máscara.
    imagem_rgba = melhorar_mascara(
        imagem_rgba
    )

    # Enquadra a fruta pela máscara.
    imagem_recortada = enquadrar_fruta(
        imagem_rgba
    )

    # Redimensiona preservando proporção.
    imagem_recortada.thumbnail(
        TAMANHO_FINAL,
        Image.Resampling.LANCZOS
    )

    # Redução de ruídos e realce.
    imagem_rgb = imagem_recortada.convert(
        "RGB"
    )

    imagem_rgb = melhorar_detalhes(
        imagem_rgb
    )

    # Cria fundo branco.
    fundo = Image.new(
        "RGB",
        TAMANHO_FINAL,
        (255, 255, 255)
    )

    x = (
        TAMANHO_FINAL[0] - imagem_rgb.width
    ) // 2

    y = (
        TAMANHO_FINAL[1] - imagem_rgb.height
    ) // 2

    # Utiliza a máscara para preservar transparência.
    fundo.paste(
        imagem_rgb,
        (x, y),
        imagem_recortada.getchannel("A")
    )

    # Ajuste de contraste.
    fundo = ImageEnhance.Contrast(
        fundo
    ).enhance(CONTRASTE)

    return fundo, imagem_recortada


# ==========================================
# SALVAMENTO DA MÁSCARA
# ==========================================

def salvar_mascara(imagem_rgba, caminho_saida):

    mascara = imagem_rgba.getchannel("A")

    mascara.save(
        caminho_saida,
        format="PNG"
    )


# ==========================================
# PROCESSAMENTO INDIVIDUAL
# ==========================================

def processar_tarefa(
    categoria,
    caminho_imagem,
    sessao
):

    pasta_img_saida = (
        PASTA_PROCESSADAS /
        categoria.name
    )

    pasta_mask_saida = (
        PASTA_MASCARAS /
        categoria.name
    )

    pasta_img_saida.mkdir(
        parents=True,
        exist_ok=True
    )

    pasta_mask_saida.mkdir(
        parents=True,
        exist_ok=True
    )

    nome_saida = (
        caminho_imagem.stem + ".png"
    )

    caminho_processada = (
        pasta_img_saida / nome_saida
    )

    caminho_mascara = (
        pasta_mask_saida / nome_saida
    )

    dados = {
        "categoria": categoria.name,
        "imagem_original": str(
            caminho_imagem.relative_to(
                PASTA_ORIGINAL
            )
        ),
        "imagem_processada": "",
        "mascara": "",
        "largura": "",
        "altura": "",
        "modelo_segmentacao": MODELO,
        "contraste": CONTRASTE,
        "morfologia": TAMANHO_MORFOLOGIA,
        "filtro_mediana": TAMANHO_FILTRO_MEDIANA,
        "realce": INTENSIDADE_REALCE,
        "status": "erro",
        "erro": ""
    }

    try:

        imagem_segmentada = segmentar_imagem(
            caminho_imagem,
            sessao
        )

        imagem_final, imagem_recortada = (
            preparar_imagem(
                imagem_segmentada
            )
        )

        imagem_final.save(
            caminho_processada,
            format="PNG"
        )

        salvar_mascara(
            imagem_recortada,
            caminho_mascara
        )

        dados.update({

            "imagem_processada": str(
                caminho_processada.relative_to(
                    PASTA_BASE
                )
            ),

            "mascara": str(
                caminho_mascara.relative_to(
                    PASTA_BASE
                )
            ),

            "largura": TAMANHO_FINAL[0],
            "altura": TAMANHO_FINAL[1],
            "status": "sucesso"

        })

        return dados

    except Exception as erro:

        dados["erro"] = str(erro)

        return dados


# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================

def main():

    if MAX_WORKERS < 1:
        raise ValueError(
            "MAX_WORKERS precisa ser pelo menos 1."
        )

    PASTA_PROCESSADAS.mkdir(
        parents=True,
        exist_ok=True
    )

    PASTA_MASCARAS.mkdir(
        parents=True,
        exist_ok=True
    )

    categorias = sorted(
        pasta
        for pasta in PASTA_ORIGINAL.iterdir()
        if pasta.is_dir()
    )

    if not categorias:
        raise RuntimeError(
            "Nenhuma pasta de categoria encontrada."
        )

    tarefas = []

    for categoria in categorias:

        imagens = listar_imagens(
            categoria
        )

        if LIMITE_POR_CATEGORIA is not None:

            imagens = imagens[
                :LIMITE_POR_CATEGORIA
            ]

        for imagem in imagens:

            tarefas.append(
                (categoria, imagem)
            )

    print(
        f"Categorias encontradas: {len(categorias)}"
    )

    print(
        f"Imagens selecionadas: {len(tarefas)}"
    )

    print(
        f"Modelo de segmentação: {MODELO}"
    )

    print(
        f"Tarefas simultâneas: {MAX_WORKERS}"
    )

    print(
        f"Destino: {PASTA_PROCESSADAS}"
    )

    print("\nCarregando modelo...")

    inicio_modelo = time.time()

    sessao = new_session(MODELO)

    print(
        f"Modelo carregado em "
        f"{time.time() - inicio_modelo:.1f} segundos."
    )

    print("\nIniciando processamento...\n")

    linhas_metadata = []

    processadas = 0
    erros = 0

    inicio = time.time()

    total = len(tarefas)

    if total == 0:

        print(
            "Nenhuma imagem encontrada."
        )

        return

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS,
        thread_name_prefix="fruta"
    ) as executor:

        futuros = {

            executor.submit(
                processar_tarefa,
                categoria,
                caminho,
                sessao
            ): caminho

            for categoria, caminho in tarefas

        }

        for indice, futuro in enumerate(
            as_completed(futuros),
            start=1
        ):

            dados = futuro.result()

            linhas_metadata.append(
                dados
            )

            if dados["status"] == "sucesso":

                processadas += 1

            else:

                erros += 1

                print(
                    f"\nERRO em {futuros[futuro]}: "
                    f"{dados['erro']}"
                )

            if (
                indice == 1
                or indice % 10 == 0
                or indice == total
            ):

                decorrido = (
                    time.time() - inicio
                )

                media = decorrido / indice

                restante = (
                    media * (total - indice)
                )

                porcentagem = (
                    indice / total
                ) * 100

                print(
                    f"[{indice}/{total}] "
                    f"{porcentagem:.1f}% | "
                    f"Sucesso: {processadas} | "
                    f"Erros: {erros} | "
                    f"Restante estimado: "
                    f"{restante / 60:.1f} min"
                )

    # ======================================
    # METADADOS
    # ======================================

    campos = [

        "categoria",
        "imagem_original",
        "imagem_processada",
        "mascara",
        "largura",
        "altura",
        "modelo_segmentacao",
        "contraste",
        "morfologia",
        "filtro_mediana",
        "realce",
        "status",
        "erro"

    ]

    linhas_metadata.sort(
        key=lambda linha: (
            linha["categoria"],
            linha["imagem_original"]
        )
    )

    with ARQUIVO_METADATA.open(
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
            linhas_metadata
        )

    # ======================================
    # RESUMO
    # ======================================

    print(
        "\n========== PROCESSAMENTO CONCLUÍDO =========="
    )

    print(
        f"Processadas com sucesso: {processadas}"
    )

    print(
        f"Erros: {erros}"
    )

    print(
        f"Tempo total: "
        f"{(time.time() - inicio) / 60:.1f} minutos"
    )

    print(
        f"Imagens: {PASTA_PROCESSADAS}"
    )

    print(
        f"Máscaras: {PASTA_MASCARAS}"
    )

    print(
        f"Metadados: {ARQUIVO_METADATA}"
    )


if __name__ == "__main__":
    main()

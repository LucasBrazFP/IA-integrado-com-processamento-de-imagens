# FrutAI — Preparação e análise das imagens

Este projeto prepara e analisa as imagens do dataset Fruits-262 para utilização em experimentos de visão computacional e treinamento de modelos de inteligência artificial.

A preparação é realizada em duas etapas principais: segmentação e pré-processamento das frutas com `main.py`; e análise das imagens segmentadas por meio da Transformada Rápida de Fourier (FFT), com `fft.py`.

## Download do Dataset

O projeto utiliza o dataset **Fruits-262**, disponibilizado publicamente no Kaggle por meio da biblioteca `kagglehub`.

Para realizar o download automático, execute o arquivo `baixar_dataset.py`:

```bash
python baixar_dataset.py
```

## 1. Estrutura esperada

O arquivo `.env` deve estar na pasta do projeto, junto com `main.py` e `fft.py`, e apontar para a pasta que contém as categorias de frutas:

```bash 
env =path_pictures=C:\Users\PC\.cache\kagglehub\datasets\aelchimminut\fruits262\versions\7\Resized_Fruits-262\208x256 
```

Dentro de `208x256`, deve existir uma pasta para cada categoria, contendo as imagens daquela fruta.

Os scripts criam as pastas de saída ao lado de `208x256`, dentro de `Resized_Fruits-262`:

```text
Resized_Fruits-262/
├── 208x256/
├── Processed_Images_Enhanced/
├── Segmentation_Masks_Enhanced/
├── FFT_Results_Enhanced/
├── metadata_enhanced.csv
├── fft_metadata_enhanced.csv
└── fft_features.csv
```

## 2. Instalação

Recomenda-se utilizar um ambiente virtual para instalar as dependências:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

As bibliotecas utilizadas diretamente por `main.py` e `fft.py` estão listadas em `requirements.txt`.

## 3. Etapa 1 — Segmentação e pré-processamento

Execute:

```powershell
python main.py
```

O script realiza as seguintes operações:

- percorre as pastas de categorias e identifica arquivos com extensões `.jpg`, `.jpeg`, `.png`, `.webp` e `.bmp`;
- utiliza `rembg` com o modelo `u2netp` para estimar o primeiro plano da imagem;
- utiliza o canal alfa retornado pela segmentação como máscara;
- aplica operações morfológicas de fechamento e abertura para tentar melhorar a máscara;
- suaviza levemente as bordas da máscara;
- identifica a região delimitada pela máscara e recorta a área da fruta, acrescentando uma margem proporcional de 8%;
- redimensiona a fruta proporcionalmente para caber em uma imagem de 300 × 300 pixels;
- aplica filtro de mediana para reduzir ruídos;
- aplica `UnsharpMask` para realçar bordas e detalhes;
- centraliza a fruta em um fundo branco;
- aplica contraste de 1,10;
- salva a imagem processada em `Processed_Images_Enhanced`;
- salva a máscara em `Segmentation_Masks_Enhanced`;
- registra categoria, caminhos, dimensões, parâmetros, status e eventuais erros em `metadata_enhanced.csv`.

### Configurações importantes do `main.py`

As configurações atuais do código enviado são:

| Configuração | Valor atual | Função |
|---|---:|---|
| `TAMANHO_FINAL` | `(300, 300)` | Dimensão final da imagem processada |
| `MODELO` | `u2netp` | Modelo utilizado pelo `rembg` |
| `CONTRASTE` | `1.10` | Ajuste de contraste |
| `MARGEM_FRUTA` | `0.08` | Margem proporcional ao redor da fruta |
| `TAMANHO_MORFOLOGIA` | `3` | Tamanho dos filtros morfológicos |
| `RAIO_SUAVIZACAO_MASCARA` | `0.5` | Suavização da máscara |
| `TAMANHO_FILTRO_MEDIANA` | `3` | Redução de ruídos |
| `RAIO_REALCE` | `1.2` | Raio do realce de detalhes |
| `INTENSIDADE_REALCE` | `110` | Intensidade do realce |
| `LIMIAR_REALCE` | `3` | Limiar do realce |
| `LIMITE_POR_CATEGORIA` | `300` | Máximo de imagens selecionadas por categoria |
| `MAX_WORKERS` | `100` | Número máximo de tarefas simultâneas solicitado ao executor |

O valor `LIMITE_POR_CATEGORIA = 300` significa que o script seleciona no máximo 300 imagens de cada categoria. Para selecionar todas as imagens, altere o valor para `None`.

O valor atual `MAX_WORKERS = 100` é elevado e pode consumir muita memória e CPU, especialmente porque a segmentação utiliza um modelo de rede neural. Recomenda-se começar com um valor menor, como `2`, e aumentar gradualmente apenas se o computador permanecer estável. Mais tarefas simultâneas não garantem maior velocidade.

**Importante:** o `u2netp` é um modelo genérico de remoção de fundo. Ele pode manter partes do cenário ou remover partes da fruta. É recomendável inspecionar imagens e máscaras de diferentes categorias antes de utilizar todo o resultado em treinamento.

## 4. Etapa 2 — Transformada Rápida de Fourier (FFT)

Execute esta etapa depois de concluir o processamento com `main.py`:

```powershell
python fft.py
```

O script:

- lê as imagens de `Processed_Images_Enhanced`;
- lê as máscaras correspondentes em `Segmentation_Masks_Enhanced`;
- converte as imagens e máscaras para escala de cinza;
- utiliza a máscara para delimitar a região de interesse da fruta;
- remove a média dos valores da região para reduzir a influência do brilho global;
- aplica uma janela Hann para reduzir efeitos artificiais nas bordas do recorte;
- calcula a FFT bidimensional com `numpy.fft.fft2`;
- centraliza o espectro com `numpy.fft.fftshift`;
- calcula a magnitude e sua representação logarítmica;
- separa as regiões de baixas e altas frequências;
- calcula energia total, energia de baixa frequência, energia de alta frequência e suas proporções;
- realiza uma análise radial dividida em 16 regiões;
- extrai características complementares, como energia média, desvio padrão do espectro, frequência dominante e área da máscara;
- salva imagens dos espectros em `FFT_Results_Enhanced`;
- registra o processamento em `fft_metadata_enhanced.csv`;
- salva as características numéricas em `fft_features.csv`.

Para cada imagem processada, são geradas representações do espectro completo, das baixas frequências e das altas frequências.

A FFT produz informações sobre a distribuição de frequências da imagem. Ela não identifica diretamente a fruta e, na versão atual, suas características numéricas são armazenadas separadamente. Elas ainda não são utilizadas diretamente como entrada da MobileNetV3.

## 5. Ordem de execução

A ordem correta é:

```text
Dataset Fruits-262
        |
        v
    baixar_dataset.py
        |
        v
      main.py
        |
        v
Segmentação e pré-processamento
        |
        +--> Processed_Images_Enhanced
        |
        +--> Segmentation_Masks_Enhanced
        |
        +--> metadata_enhanced.csv
        |
        v
      fft.py
        |
        +--> FFT_Results_Enhanced
        |
        +--> fft_metadata_enhanced.csv
        |
        +--> fft_features.csv
```

O `fft.py` depende das imagens e máscaras produzidas pelo `main.py`. Portanto, execute primeiro `main.py` e depois `fft.py`.

## 6. Cuidados para o treinamento futuro

- Manter os nomes das pastas de categorias originais para preservar os rótulos (*labels*).
- Separar os dados de treinamento, validação e teste antes de ajustar parâmetros com base nos resultados.
- Evitar que imagens duplicadas ou quase idênticas apareçam em conjuntos diferentes.
- Inspecionar manualmente amostras das imagens e máscaras para identificar falhas de segmentação.
- Não assumir que o ajuste de contraste, a remoção de fundo ou a FFT melhoraram a classificação sem comparar resultados experimentais.
- Para uma CNN como a MobileNetV3, as imagens de `Processed_Images_Enhanced` podem ser utilizadas como entrada principal.
- Os dados de `fft_features.csv` podem ser avaliados em experimentos separados ou como características adicionais, desde que essa integração seja implementada e testada.

## 7. Limitações e observações técnicas

- A segmentação é realizada por um modelo genérico (`u2netp`), e não por um modelo treinado especificamente para as categorias do Fruits-262.
- A qualidade das máscaras depende das imagens de entrada e do comportamento do modelo de segmentação.
- O paralelismo atual está configurado em `MAX_WORKERS = 100`. Esse valor pode sobrecarregar o computador; ajuste-o conforme a memória e a capacidade de processamento disponíveis.
- O limite atual é de 300 imagens por categoria. Para processar todas, é necessário definir `LIMITE_POR_CATEGORIA = None`.
- No `main.py` enviado, a imagem final é salva em 300 × 300, mas a máscara é salva a partir da imagem recortada. Assim, a máscara pode ter dimensões diferentes da imagem final.
- O `fft.py` redimensiona a máscara quando as dimensões não coincidem. Esse redimensionamento ajusta o tamanho, mas pode não preservar o mesmo enquadramento e alinhamento espacial da fruta. Para uma análise mais confiável, o ideal é que o `main.py` salve uma máscara final de 300 × 300, posicionada no mesmo local da fruta na imagem processada.
- A FFT depende da região selecionada pela máscara, do recorte e do pré-processamento. O espectro não garante, por si só, a identificação correta da fruta.
- As características da FFT são salvas em CSV, mas não estão integradas ao classificador nesta etapa.
- As pastas de saída são criadas ao lado da pasta `208x256`. Verifique o espaço disponível em disco antes de processar o dataset completo.
- A primeira execução do `rembg` pode exigir o download dos arquivos do modelo `u2netp`.

## 8. Arquivos gerados

| Arquivo ou pasta | Conteúdo |
|---|---|
| `Processed_Images_Enhanced` | Imagens segmentadas e pré-processadas em PNG |
| `Segmentation_Masks_Enhanced` | Máscaras geradas a partir da segmentação |
| `metadata_enhanced.csv` | Registro do processamento de cada imagem |
| `FFT_Results_Enhanced` | Espectros completos, de baixas e de altas frequências |
| `fft_metadata_enhanced.csv` | Status e caminhos dos resultados da FFT |
| `fft_features.csv` | Características numéricas extraídas dos espectros |

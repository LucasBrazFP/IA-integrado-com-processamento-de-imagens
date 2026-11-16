# Frutix — Identificação e comparação inteligente de frutas

O **Frutix** é um projeto de identificação de frutas por imagem. Ele utiliza uma rede neural treinada com o dataset **Fruits-262** e uma segunda avaliação feita por um modelo de IA generativa acessado pela API da Groq.

O aplicativo compara as duas avaliações por meio de uma pontuação ponderada, dando maior peso à MobileNetV3. O projeto também possui etapas de preparação das imagens, análise complementar por Transformada Rápida de Fourier (FFT), treinamento e geração de relatórios.

> **Importante:** a Groq é a plataforma/provedora de API utilizada para acessar o modelo Qwen. Qwen é o modelo de IA generativa; não deve ser confundido com Grok.

---

## 1. Tecnologias utilizadas

- **Python:** linguagem principal do projeto.
- **PyTorch e Torchvision:** treinamento e execução da rede neural.
- **MobileNetV3 Small:** modelo de classificação de imagens utilizado como IA treinada.
- **Fruits-262:** conjunto de dados de imagens de frutas.
- **KaggleHub:** download do dataset.
- **rembg / U2NetP:** remoção do fundo e segmentação da fruta.
- **Pillow:** leitura e processamento de imagens.
- **NumPy:** operações numéricas e cálculo da FFT.
- **Pandas:** leitura e análise dos relatórios CSV.
- **Groq API + Qwen:** segunda avaliação visual por IA generativa.
- **Tkinter:** interface gráfica do aplicativo.

---

## 2. Estrutura do projeto

Uma organização sugerida para os arquivos é:

```text
Frutix/
│
├── baixar_dataset.py
├── verificar_dataset.py
├── main.py
├── fft.py
├── train.py
├── analise.py
├── app.py
│
├── .env
├── .env.example
├── requirements.txt
├── README.md
├── iconeap.ico
│
├── modelo/
│   └── fruits262_mobilenet_enhanced.pth
│
├── resultados/
│   ├── relatorio_classes_enhanced.csv
│   ├── matriz_confusao_enhanced.csv
│   └── confianca_previsoes.csv
│
├── Processed_Images_Enhanced/
├── Segmentation_Masks_Enhanced/
├── FFT_Results_Enhanced/
│
├── metadata_enhanced.csv
├── fft_metadata_enhanced.csv
└── fft_features.csv
```

As pastas e os arquivos de resultados são criados durante a execução das etapas correspondentes. O arquivo de modelo `.pth` é gerado pelo treinamento. Não é necessário manter cópias do dataset inteiro dentro da pasta do código, desde que o caminho esteja configurado corretamente.

---

## 3. Instalação

Recomenda-se utilizar um ambiente virtual.

No terminal, dentro da pasta do projeto:

```bash
python -m venv venv
```

Ative o ambiente virtual no Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**PyTorch:** caso seja necessário utilizar a GPU NVIDIA, instale uma versão do PyTorch compatível com a versão do CUDA e os drivers instalados no computador. Consulte a página oficial de instalação do PyTorch para selecionar o comando adequado. Se não houver GPU compatível, o projeto pode utilizar a CPU, embora o treinamento possa demorar mais.

---

## 4. Configuração do arquivo `.env`

Crie um arquivo chamado `.env` na pasta principal do projeto. Ele armazena os caminhos do dataset e as configurações de acesso à API da Groq.

Exemplo:

```env
# Caminho utilizado pelo pré-processamento das imagens
path_pictures=C:\Users\SEU_USUARIO\.cache\kagglehub\datasets\aelchimminut\fruits262\versions\7\Resized_Fruits-262\208x256

# Caminho utilizado pelo treinamento e pela verificação do dataset
DATASET_PATH=C:\Users\SEU_USUARIO\.cache\kagglehub\datasets\aelchimminut\fruits262\versions\7\Resized_Fruits-262\Processed_Images_Enhanced

# Credenciais e modelo da API Groq
GROQ_API_KEY=SUA_CHAVE_DA_GROQ
GROQ_MODEL=qwen/qwen3.8-27b
```

Substitua `SEU_USUARIO` pelo nome da pasta do usuário do Windows e ajuste os caminhos conforme o local em que o dataset foi baixado e processado.

- `path_pictures`: aponta para a pasta original `208x256`, utilizada pelo `main.py`.
- `DATASET_PATH`: aponta para `Processed_Images_Enhanced`, utilizada pelo `train.py` e pelo `verificar_dataset.py`.
- `GROQ_API_KEY`: chave privada de acesso à API da Groq.
- `GROQ_MODEL`: identificador do modelo utilizado pela aplicação. Confirme na documentação da Groq se o identificador está disponível para sua conta e se o modelo aceita entrada de imagem.

**Segurança:** não compartilhe sua chave da API e não envie o arquivo `.env` para o GitHub. Inclua `.env` no `.gitignore`. Para compartilhar o projeto, disponibilize um `.env.example` sem credenciais reais.

Exemplo de `.env.example`:

```env
path_pictures=C:\CAMINHO\DO\DATASET\Resized_Fruits-262\208x256
DATASET_PATH=C:\CAMINHO\DO\DATASET\Processed_Images_Enhanced
GROQ_API_KEY=COLOQUE_SUA_CHAVE_AQUI
GROQ_MODEL=qwen/qwen3.8-27b
```

---

## 5. Download do dataset

O projeto utiliza o dataset **Fruits-262**, disponibilizado no Kaggle.

Para baixar o conjunto de dados automaticamente, execute:

```bash
python baixar_dataset.py
```

O script utiliza a biblioteca `kagglehub`:

```python
import kagglehub

path = kagglehub.dataset_download("aelchimminut/fruits262")

print("Path to dataset files:", path)
```

Após o download, o terminal informa o caminho em que o dataset foi armazenado. Utilize esse caminho para localizar a pasta `Resized_Fruits-262\208x256` e configurar a variável `path_pictures` no `.env`.

O download é realizado uma vez por ambiente, salvo quando for necessário baixar novamente ou utilizar outra versão do dataset.

---

## 6. Verificação do dataset

Depois de baixar o dataset e antes de iniciar o pré-processamento ou treinamento, é possível executar:

```bash
python verificar_dataset.py
```

Esse script auxiliar lê `DATASET_PATH` no `.env` e apresenta:

- quantidade de categorias encontradas;
- quantidade total de imagens;
- categorias com menor quantidade de imagens;
- categorias sem imagens;
- categorias que possuem somente uma imagem.

O `verificar_dataset.py` **não realiza treinamento nem altera as imagens**. Sua finalidade é facilitar a conferência da organização e da quantidade de arquivos do dataset.

---

## 7. Pré-processamento das imagens

Com o caminho `path_pictures` configurado, execute:

```bash
python main.py
```

O `main.py` prepara as imagens para as próximas etapas. O processamento inclui:

1. leitura das imagens originais;
2. remoção do fundo utilizando `rembg` com o modelo U2NetP;
3. geração e tratamento de máscaras de segmentação;
4. recorte e centralização da fruta;
5. redimensionamento para o padrão definido no script;
6. aplicação de ajustes de imagem;
7. salvamento das imagens processadas, máscaras e metadados.

As principais saídas da versão aprimorada são:

- `Processed_Images_Enhanced/`
- `Segmentation_Masks_Enhanced/`
- `metadata_enhanced.csv`

O script possui um limite de processamento por categoria configurado no próprio código. Confira o valor de `LIMITE_POR_CATEGORIA` antes de executar. Para processar todas as imagens de cada categoria, o código deve estar configurado para não aplicar limite.

**Observação de desempenho:** o valor de `MAX_WORKERS` configurado no script influencia o uso de memória e processador. Um número muito alto pode deixar o computador lento ou causar falta de memória. Ajuste esse valor de acordo com o hardware disponível.

---

## 8. Análise de frequência (FFT)

Após o pré-processamento, execute:

```bash
python fft.py
```

O `fft.py` utiliza as imagens processadas e suas máscaras para realizar uma análise complementar por Transformada Rápida de Fourier (FFT).

O script calcula informações relacionadas à distribuição de frequências da imagem, como:

- energia de baixas e altas frequências;
- energia espectral;
- frequência dominante;
- distribuição radial de frequências;
- características derivadas da máscara da fruta.

As saídas incluem:

- `FFT_Results_Enhanced/`
- `fft_metadata_enhanced.csv`
- `fft_features.csv`

**Importante:** nesta versão do projeto, a FFT é uma etapa de análise complementar. As características geradas em `fft_features.csv` não são utilizadas diretamente como entrada da MobileNetV3 no treinamento descrito neste README.

---

## 9. Treinamento da IA — MobileNetV3

Antes de treinar, confirme se `DATASET_PATH` aponta para a pasta `Processed_Images_Enhanced` e se as imagens estão organizadas em subpastas, uma para cada categoria de fruta.

Execute:

```bash
python train.py
```

O treinamento utiliza a arquitetura **MobileNetV3 Small**, com pesos pré-treinados, adaptando a camada final para as categorias encontradas no dataset.

O `train.py` realiza as seguintes etapas:

1. carrega as imagens organizadas por classe;
2. divide os dados em conjuntos de treinamento e validação, separando as imagens por categoria;
3. aplica redimensionamento e aumento de dados (*data augmentation*) no treinamento;
4. testa três configurações de hiperparâmetros;
5. realiza o treinamento final com AdamW e ajuste da taxa de aprendizado;
6. utiliza `WeightedRandomSampler` e atualização de pesos das amostras para dar mais atenção a exemplos com maior perda;
7. aplica parada antecipada (*early stopping*) quando não há melhora na perda de validação;
8. salva o melhor checkpoint e gera relatórios de avaliação.

Entre os parâmetros definidos no script estão:

- tamanho de entrada: `224 × 224`;
- lote (*batch size*): `16`;
- máximo de épocas do treinamento final: `40`;
- paciência para parada antecipada: `8` épocas;
- limiar utilizado para marcar uma previsão como de maior confiança: `70%`.

A busca de hiperparâmetros testa três configurações por cinco épocas cada, antes do treinamento final. Por isso, o processo pode levar algum tempo, especialmente em computadores sem GPU.

### Arquivo do modelo

Ao finalizar, o script salva o modelo em:

```text
modelo/fruits262_mobilenet_enhanced.pth
```

Esse arquivo contém os pesos treinados, a lista de classes e informações necessárias para carregar o modelo posteriormente.

**Não é necessário treinar novamente em outro computador** se o arquivo `.pth` for disponibilizado junto ao projeto e for compatível com o código e as dependências. O computador que executará o aplicativo precisa ter as bibliotecas necessárias e o arquivo do modelo no caminho esperado.

### Relatórios gerados

O treinamento cria a pasta `resultados/` com:

- `relatorio_classes_enhanced.csv`: métricas por categoria, incluindo quantidade de imagens de validação, acertos, acurácia, precisão, recall e F1-score;
- `matriz_confusao_enhanced.csv`: matriz que mostra as classes reais e as classes previstas;
- `confianca_previsoes.csv`: classe real, classe prevista, confiança estimada e indicação de confiança da previsão.

Os valores efetivos de desempenho devem ser obtidos pela execução do treinamento. Não devem ser presumidos ou preenchidos sem consultar os relatórios gerados.

---

## 10. Análise simplificada dos resultados

O arquivo `analise.py` é um script auxiliar de leitura dos relatórios CSV. Ele calcula um resumo geral e lista exemplos de confusões entre classes.

Ele não treina a rede neural, não modifica o modelo e não é necessário para abrir o aplicativo.

**Atenção à compatibilidade dos nomes:** o `train.py` apresentado neste projeto gera `relatorio_classes_enhanced.csv` e `matriz_confusao_enhanced.csv`, enquanto o `analise.py` fornecido procura `relatorio_classes.csv` e `matriz_confusao.csv`. Para executar a análise, ajuste no `analise.py` os nomes dos arquivos para os nomes `*_enhanced.csv` gerados pelo treinamento, ou faça uma cópia dos relatórios com os nomes esperados pelo script.

Os scripts `verificar_dataset.py` e `analise.py` são ferramentas auxiliares, elaboradas com apoio de IA para facilitar a verificação do dataset e a leitura dos resultados de maneira simples. Eles não fazem parte do fluxo essencial de treinamento e execução do Frutix e podem ser utilizados apenas quando for conveniente.

---

## 11. Execução do aplicativo Frutix

Para abrir a interface gráfica, execute:

```bash
python app.py
```

O aplicativo utiliza o checkpoint disponível em:

```text
modelo/fruits262_mobilenet_enhanced.pth
```

Se esse arquivo não existir, o código também procura o modelo antigo:

```text
modelo/fruits262_mobilenet.pth
```

O arquivo `iconeap.ico` é utilizado como ícone da janela e deve estar na pasta principal do projeto, conforme o caminho definido no `app.py`.

### Como funciona a identificação

1. O usuário seleciona uma imagem de fruta.
2. A imagem passa pela etapa de preparação e segmentação com U2NetP.
3. A MobileNetV3 analisa a imagem preparada e apresenta sua classe prevista e confiança estimada.
4. A imagem original é enviada à API da Groq, que utiliza o modelo Qwen configurado para fornecer uma segunda avaliação visual.
5. A resposta da IA generativa é limitada às classes existentes no checkpoint do Fruits-262.
6. O aplicativo compara as duas avaliações e apresenta o resultado na interface.

### Balança de decisão

A comparação utiliza os seguintes pesos definidos no `app.py`:

- MobileNetV3: **65%**;
- Qwen via Groq: **35%**.

A pontuação considera o peso de cada IA e a confiança informada por ela. Assim, a MobileNetV3 tem maior influência na decisão final.

**Importante sobre as porcentagens:** a confiança informada pelo Qwen é uma estimativa fornecida pelo próprio modelo, não uma medida calibrada. A pontuação exibida pela balança é uma pontuação relativa calculada pelo aplicativo, não uma probabilidade estatística de que a fruta esteja correta.

A aplicação depende de conexão com a internet e de uma chave válida da API da Groq para realizar a segunda avaliação. Se a API não estiver configurada ou disponível, a comparação entre as duas IAs não poderá ser concluída.

---

## 12. Fluxo geral de execução

A sequência recomendada para preparar o projeto do zero é:

```text
1. Instalar as dependências
          |
          v
2. Configurar o arquivo .env
          |
          v
3. Baixar o Fruits-262
          |
          v
4. Verificar a organização do dataset (opcional)
          |
          v
5. Pré-processar as imagens (main.py)
          |
          v
6. Realizar a análise FFT (fft.py)
          |
          v
7. Treinar a MobileNetV3 (train.py)
          |
          v
8. Consultar os relatórios (analise.py - opcional)
          |
          v
9. Executar o aplicativo (app.py)
```

Se o dataset já estiver baixado e processado e o checkpoint `.pth` já existir, não é necessário repetir as etapas de download, pré-processamento e treinamento para simplesmente executar o aplicativo.

---

## 13. Arquivos essenciais e auxiliares

| Arquivo | Finalidade |
|---|---|
| `baixar_dataset.py` | Baixar o Fruits-262 pelo KaggleHub. |
| `main.py` | Pré-processar imagens e gerar máscaras. |
| `fft.py` | Calcular características de frequência das imagens. |
| `train.py` | Treinar a MobileNetV3 e gerar o checkpoint e relatórios. |
| `app.py` | Executar a interface gráfica e comparar as duas IAs. |
| `verificar_dataset.py` | Auxiliar na conferência de categorias e quantidade de imagens. |
| `analise.py` | Auxiliar na leitura e resumo dos relatórios CSV. |
| `.env` | Armazenar caminhos e configurações privadas da API. |
| `requirements.txt` | Listar as bibliotecas necessárias para o projeto. |

Os arquivos `baixar_dataset.py`, `main.py`, `train.py` e `app.py` participam do fluxo principal de preparação, treinamento e utilização. O `fft.py` realiza uma análise complementar. Já `verificar_dataset.py` e `analise.py` são utilitários opcionais.

---

## 14. Limitações e observações

- A qualidade da classificação depende da variedade, quantidade e qualidade das imagens do Fruits-262 e do treinamento realizado.
- Imagens com fundo complexo, iluminação inadequada, fruta parcialmente escondida ou ângulo incomum podem prejudicar a identificação.
- A remoção de fundo pode apresentar falhas em imagens com objetos próximos ou contornos pouco definidos.
- A IA generativa pode fornecer uma classificação incorreta ou uma estimativa de confiança imprecisa.
- A pontuação da balança é uma comparação ponderada, não uma garantia de acerto.
- A FFT gera informações complementares, mas suas características não estão integradas diretamente à entrada da MobileNetV3 nesta versão.
- O aplicativo depende da disponibilidade da API da Groq para obter a segunda opinião.
- O identificador do modelo configurado em `GROQ_MODEL` deve ser válido e compatível com análise de imagens na API utilizada.

---

## 15. Compartilhamento do projeto

Para que outra pessoa consiga executar o Frutix, disponibilize:

- os arquivos `.py` do projeto;
- `requirements.txt`;
- `README.md`;
- o checkpoint treinado `modelo/fruits262_mobilenet_enhanced.pth`;
- o ícone `iconeap.ico`, se utilizado;
- os relatórios CSV, caso queira apresentar os resultados do treinamento;
- um `.env.example` sem chave real.

O dataset pode ser baixado pelo próprio `baixar_dataset.py`, evitando a necessidade de enviar uma cópia de vários gigabytes.

Cada pessoa deverá configurar seu próprio arquivo `.env` e utilizar sua própria chave da API da Groq. **Nunca compartilhe sua chave privada.**

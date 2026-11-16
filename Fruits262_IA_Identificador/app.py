import os
import io
import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
from rembg import new_session, remove
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


load_dotenv()

MODEL_PATHS = [
    Path("modelo/fruits262_mobilenet_enhanced.pth"),
    Path("modelo/fruits262_mobilenet.pth")
]

IMAGE_SIZE = 224
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

PESO_MOBILENET = 0.70
PESO_GROQ = 0.30


def traduzir_fruta(nome):
    nome = nome.replace("_", " ").replace("-", " ").strip()

    traducoes = {
        "apple": "Maçã",
        "banana": "Banana",
        "orange": "Laranja",
        "grape": "Uva",
        "watermelon": "Melancia",
        "strawberry": "Morango",
        "pineapple": "Abacaxi",
        "mango": "Manga",
        "papaya": "Mamão",
        "pear": "Pera",
        "peach": "Pêssego",
        "plum": "Ameixa",
        "cherry": "Cereja",
        "lemon": "Limão",
        "lime": "Limão-verde",
        "kiwi": "Kiwi",
        "coconut": "Coco",
        "avocado": "Abacate",
        "guava": "Goiaba",
        "passion fruit": "Maracujá",
        "dragon fruit": "Pitaya",
        "pitaya": "Pitaya",
        "pomegranate": "Romã",
        "fig": "Figo",
        "apricot": "Damasco",
        "blueberry": "Mirtilo",
        "raspberry": "Framboesa",
        "blackberry": "Amora-preta",
        "grapefruit": "Toranja",
        "mandarin": "Mexerica",
        "tangerine": "Tangerina",
        "persimmon": "Caqui",
        "lychee": "Lichia",
        "durian": "Durião",
        "jackfruit": "Jaca",
        "star fruit": "Carambola",
        "carambola": "Carambola",
        "soursop": "Graviola",
        "acerola": "Acerola",
        "acai": "Açaí",
        "jabuticaba": "Jabuticaba",
        "tomato": "Tomate",
        "cucumber": "Pepino",
        "pumpkin": "Abóbora",
        "corn": "Milho",
        "melon": "Melão",
        "cantaloupe": "Melão-cantalupo",
        "honeydew": "Melão",
        "physalis": "Physalis",
        "cashew": "Caju",
        "olive": "Azeitona",
    }
    return traducoes.get(nome.lower(), nome.title())


class FruitApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Frutix | Comparador de Inteligência Artificial")
        self.root.geometry("1200x850")
        self.root.minsize(1000, 720)
        self.root.configure(bg="#F3F6FA")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.session = None
        self.photo = None
        self.current_image = None
        self.analisando = False
        self.model = None
        self.classes = []
        self.transform = None
        self.client = None
        self.carregar_modelo()
        self.configurar_groq()
        self.criar_interface()



    def carregar_modelo(self):

        model_path = next(
            (path for path in MODEL_PATHS if path.exists()),
            None
        )

        if model_path is None:
            raise FileNotFoundError(
                "Nenhum modelo foi encontrado.\n\n"
                "Procure por:\n"
                + "\n".join(str(path) for path in MODEL_PATHS)
            )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        self.classes = checkpoint["classes"]

        self.model = models.mobilenet_v3_small(
            weights=None
        )

        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features,
            len(self.classes)
        )

        self.model.load_state_dict(
            checkpoint["model_state"]
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                checkpoint["normalization_mean"],
                checkpoint["normalization_std"]
            )
        ])

        self.model_path = model_path

        print("Modelo carregado:", model_path)
        print("Classes disponíveis:", len(self.classes))

    

    def configurar_groq(self):

        if GROQ_API_KEY and OpenAI:

            self.client = OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
                timeout=90
            )

   
    def criar_interface(self):

        header = tk.Frame(
            self.root,
            bg="#172554",
            height=105
        )

        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Frutix",
            font=("Arial", 27, "bold"),
            fg="white",
            bg="#172554"
        ).pack(anchor="w", padx=30, pady=(17, 0))

        tk.Label(
            header,
            text="Comparação inteligente de identificações de frutas",
            font=("Arial", 11),
            fg="#BFDBFE",
            bg="#172554"
        ).pack(anchor="w", padx=32)

        main = tk.Frame(
            self.root,
            bg="#F3F6FA"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=20
        )

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        

        image_card = self.criar_card(main)
        image_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 12)
        )

        tk.Label(
            image_card,
            text="Imagem analisada",
            font=("Arial", 16, "bold"),
            bg="white",
            fg="#172033"
        ).pack(anchor="w", padx=20, pady=(20, 12))

        self.image_area = tk.Label(
            image_card,
            text="Selecione uma imagem\npara iniciar a análise",
            bg="#F1F5F9",
            fg="#64748B",
            font=("Arial", 12),
            justify="center"
        )

        self.image_area.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        tk.Button(
            image_card,
            text="Selecionar imagem",
            command=self.selecionar_imagem,
            bg="#2563EB",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="flat",
            cursor="hand2",
            pady=12
        ).pack(
            fill="x",
            padx=20,
            pady=(10, 20)
        )

        self.device_label = tk.Label(
            image_card,
            text=f"Modelo: {self.model_path.name}\nDispositivo: {self.device}",
            bg="white",
            fg="#64748B",
            font=("Arial", 9),
            justify="left"
        )

        self.device_label.pack(
            anchor="w",
            padx=20,
            pady=(0, 15)
        )


        results = tk.Frame(
            main,
            bg="#F3F6FA"
        )

        results.grid(
            row=0,
            column=1,
            sticky="nsew"
        )


        opinions = tk.Frame(
            results,
            bg="#F3F6FA"
        )

        opinions.pack(
            fill="x"
        )

        opinions.columnconfigure(0, weight=1)
        opinions.columnconfigure(1, weight=1)

        self.mobile_card = self.criar_card(opinions)
        self.mobile_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 7)
        )

        self.groq_card = self.criar_card(opinions)
        self.groq_card.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(7, 0)
        )

        self.criar_painel_ia()
        self.criar_painel_groq()


        balance_card = self.criar_card(results)

        balance_card.pack(
            fill="both",
            expand=True,
            pady=(15, 0)
        )

        tk.Label(
            balance_card,
            text="Balança de decisão",
            font=("Arial", 17, "bold"),
            bg="white",
            fg="#172033"
        ).pack(
            anchor="w",
            padx=20,
            pady=(18, 5)
        )

        tk.Label(
            balance_card,
            text="Comparação das avaliações das duas inteligências",
            font=("Arial", 10),
            bg="white",
            fg="#64748B"
        ).pack(
            anchor="w",
            padx=20
        )

        self.balance_canvas = tk.Canvas(
            balance_card,
            height=145,
            bg="white",
            highlightthickness=0
        )

        self.balance_canvas.pack(
            fill="x",
            padx=20,
            pady=5
        )

        self.balance_canvas.bind(
            "<Configure>",
            lambda event: self.desenhar_balanca()
        )

        self.consensus_label = tk.Label(
            balance_card,
            text="Aguardando análise das IAs...",
            font=("Arial", 15, "bold"),
            bg="white",
            fg="#334155",
            wraplength=650
        )

        self.consensus_label.pack(
            pady=(0, 5)
        )

        self.score_label = tk.Label(
            balance_card,
            text="",
            font=("Arial", 11),
            bg="white",
            fg="#64748B"
        )

        self.score_label.pack(
            pady=(0, 15)
        )

        self.status_label = tk.Label(
            self.root,
            text="Pronto para analisar.",
            font=("Arial", 10),
            bg="#F3F6FA",
            fg="#64748B"
        )

        self.status_label.pack(pady=(0, 12))

    def criar_card(self, parent):

        return tk.Frame(
            parent,
            bg="white",
            highlightbackground="#E2E8F0",
            highlightthickness=1
        )

    def criar_painel_ia(self):

        tk.Label(
            self.mobile_card,
            text="SUA IA TREINADA",
            font=("Arial", 10, "bold"),
            fg="#2563EB",
            bg="white"
        ).pack(anchor="w", padx=18, pady=(18, 5))

        tk.Label(
            self.mobile_card,
            text="MobileNetV3",
            font=("Arial", 15, "bold"),
            fg="#172033",
            bg="white"
        ).pack(anchor="w", padx=18)

        self.mobile_result = tk.Label(
            self.mobile_card,
            text="Aguardando...",
            font=("Arial", 17, "bold"),
            fg="#2563EB",
            bg="white",
            wraplength=260
        )

        self.mobile_result.pack(
            anchor="w",
            padx=18,
            pady=(18, 5)
        )

        self.mobile_confidence = tk.Label(
            self.mobile_card,
            text="Confiança: --",
            font=("Arial", 11),
            fg="#64748B",
            bg="white"
        )

        self.mobile_confidence.pack(
            anchor="w",
            padx=18,
            pady=(0, 18)
        )

    def criar_painel_groq(self):

        tk.Label(
            self.groq_card,
            text="SEGUNDA OPINIÃO",
            font=("Arial", 10, "bold"),
            fg="#7C3AED",
            bg="white"
        ).pack(anchor="w", padx=18, pady=(18, 5))

        tk.Label(
            self.groq_card,
            text="Grok / Groq",
            font=("Arial", 15, "bold"),
            fg="#172033",
            bg="white"
        ).pack(anchor="w", padx=18)

        self.groq_result = tk.Label(
            self.groq_card,
            text="Aguardando...",
            font=("Arial", 17, "bold"),
            fg="#7C3AED",
            bg="white",
            wraplength=260
        )

        self.groq_result.pack(
            anchor="w",
            padx=18,
            pady=(18, 5)
        )

        self.groq_confidence = tk.Label(
            self.groq_card,
            text="Confiança: --",
            font=("Arial", 11),
            fg="#64748B",
            bg="white"
        )

        self.groq_confidence.pack(
            anchor="w",
            padx=18,
            pady=(0, 18)
        )


    def desenhar_balanca(self, diferenca=0):

        canvas = self.balance_canvas
        canvas.delete("all")

        largura = canvas.winfo_width()

        if largura < 100:
            return

        centro = largura / 2
        topo = 18

        inclinacao = max(-22, min(22, diferenca * 0.35))

        x_esquerda = centro - 145
        x_direita = centro + 145

        y_esquerda = topo + inclinacao
        y_direita = topo - inclinacao

        canvas.create_line(
            centro,
            35,
            centro,
            115,
            fill="#64748B",
            width=5
        )

        canvas.create_polygon(
            centro - 35, 115,
            centro + 35, 115,
            centro, 80,
            fill="#94A3B8"
        )

        canvas.create_line(
            x_esquerda,
            y_esquerda,
            x_direita,
            y_direita,
            fill="#334155",
            width=7,
            capstyle=tk.ROUND
        )

        for x, y, cor in [
            (x_esquerda, y_esquerda, "#2563EB"),
            (x_direita, y_direita, "#7C3AED")
        ]:

            canvas.create_line(
                x, y,
                x - 45, y + 48,
                fill=cor,
                width=2
            )

            canvas.create_line(
                x, y,
                x + 45, y + 48,
                fill=cor,
                width=2
            )

            canvas.create_line(
                x - 45, y + 48,
                x + 45, y + 48,
                fill=cor,
                width=3
            )

        canvas.create_text(
            x_esquerda,
            132,
            text="SUA IA",
            fill="#2563EB",
            font=("Arial", 10, "bold")
        )

        canvas.create_text(
            x_direita,
            132,
            text="GROK",
            fill="#7C3AED",
            font=("Arial", 10, "bold")
        )


    def selecionar_imagem(self):

        path = filedialog.askopenfilename(
            title="Selecione uma fruta",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.webp *.bmp")
            ]
        )

        if path:
            self.analisar(path)

    def preparar_imagem(self, path):

        imagem = Image.open(path).convert("RGB")

        if self.session is None:
            self.session = new_session("u2netp")

        buffer = io.BytesIO()
        imagem.save(buffer, format="PNG")

        resultado = remove(
            buffer.getvalue(),
            session=self.session
        )

        fruta = Image.open(
            io.BytesIO(resultado)
        ).convert("RGBA")

        alpha = fruta.getchannel("A")
        caixa = alpha.getbbox()

        if caixa:
            fruta = fruta.crop(caixa)

        fruta.thumbnail(
            (300, 300),
            Image.Resampling.LANCZOS
        )

        fundo = Image.new(
            "RGB",
            (300, 300),
            "white"
        )

        x = (300 - fruta.width) // 2
        y = (300 - fruta.height) // 2

        fundo.paste(
            fruta,
            (x, y),
            fruta.getchannel("A")
        )

        return fundo

    def analisar_mobilenet(self, imagem):

        tensor = self.transform(
            imagem
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            saidas = self.model(tensor)

            probabilidades = torch.softmax(
                saidas,
                dim=1
            )[0]

            valores, indices = torch.topk(
                probabilidades,
                k=min(5, len(self.classes))
            )

        resultado = {
            "classe": self.classes[indices[0].item()],
            "confianca": valores[0].item(),
            "probabilidades": {
                self.classes[index.item()]: value.item()
                for value, index in zip(valores, indices)
            }
        }

        return resultado


    def analisar_groq(self, path):

        if self.client is None:
            raise RuntimeError(
                "Groq não configurado. Verifique GROQ_API_KEY "
                "e a instalação da biblioteca openai."
            )

        with Image.open(path) as original:
            imagem = original.convert("RGB")
            imagem.thumbnail((1200, 1200), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            imagem.save(buffer, format="JPEG", quality=85)

        imagem_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        classes_disponiveis = [
            {
                "id": i,
                "nome": classe
            }
            for i, classe in enumerate(self.classes)
        ]

        prompt = f"""
    Você é um classificador visual de frutas especializado
    EXCLUSIVAMENTE no dataset Fruits-262.

    Sua tarefa é identificar a fruta da imagem.

    REGRAS OBRIGATÓRIAS:

    1. Escolha exclusivamente uma classe da lista fornecida.
    2. Utilize o ID numérico da classe.
    3. Nunca invente frutas ou categorias.
    4. Não substitua uma classe específica por outra mais genérica.
    5. Não utilize conhecimentos externos para criar categorias.
    6. Se estiver em dúvida, escolha a classe mais provável
    DENTRO DA LISTA.
    7. A confiança deve ser estimada entre 0 e 100.
    8. Retorne somente JSON válido.
    9. O ID precisa existir na lista de classes.

    CLASSES PERMITIDAS:
    {json.dumps(classes_disponiveis, ensure_ascii=False)}

    FORMATO OBRIGATÓRIO:

    {{
        "classe_id": 125,
        "confianca": 85,
        "justificativa": "Características visuais identificadas."
    }}
    """

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um classificador de imagens. "
                        "Só pode escolher IDs existentes no dataset "
                        "Fruits-262. Nunca invente classes. "
                        "Responda somente JSON válido."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64,"
                                    + imagem_base64
                                )
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=300
        )

        conteudo = response.choices[0].message.content

        inicio = conteudo.find("{")
        fim = conteudo.rfind("}")

        if inicio == -1 or fim == -1:
            raise ValueError(
                "O Groq não retornou um JSON válido."
            )

        dados = json.loads(conteudo[inicio:fim + 1])

        classe_id = dados.get("classe_id")

        if isinstance(classe_id, bool):
            raise ValueError("ID de classe inválido.")

        try:
            classe_id = int(classe_id)
        except (TypeError, ValueError):
            raise ValueError("O Groq retornou um ID inválido.")

        if classe_id < 0 or classe_id >= len(self.classes):
            raise ValueError(
                "O Groq tentou selecionar uma classe inexistente "
                "no dataset Fruits-262."
            )

        classe = self.classes[classe_id]

        confianca = float(dados.get("confianca", 0))

        confianca = max(0, min(100, confianca)) / 100

        return {
            "classe": classe,
            "confianca": confianca,
            "justificativa": dados.get(
                "justificativa",
                "Sem justificativa fornecida."
            )
        }



    def comparar_resultados(self, resultado_ia, resultado_groq):

        classe_ia = resultado_ia["classe"]
        classe_groq = resultado_groq["classe"]

        confianca_ia = resultado_ia["confianca"]
        confianca_groq = resultado_groq["confianca"]

        votos = {}

        votos[classe_ia] = (
            votos.get(classe_ia, 0)
            + PESO_MOBILENET * confianca_ia
        )

        votos[classe_groq] = (
            votos.get(classe_groq, 0)
            + PESO_GROQ * confianca_groq
        )

        total = sum(votos.values())

        if total == 0:
            raise ValueError("As duas IAs retornaram confiança zero.")

        pontuacoes = {
            classe: valor / total * 100
            for classe, valor in votos.items()
        }

        vencedora = max(
            pontuacoes,
            key=pontuacoes.get
        )

        empate = (
            len(pontuacoes) == 2
            and abs(
                pontuacoes[classe_ia]
                - pontuacoes[classe_groq]
            ) < 0.01
        )

        concordaram = classe_ia == classe_groq

        return {
            "pontuacoes": pontuacoes,
            "vencedora": vencedora,
            "concordaram": concordaram,
            "empate": empate
        }

    def analisar(self, path):

        if self.analisando:
            return

        self.analisando = True

        self.mobile_result.config(
            text="Analisando..."
        )

        self.groq_result.config(
            text="Analisando..."
        )

        self.mobile_confidence.config(
            text="Aguarde..."
        )

        self.groq_confidence.config(
            text="Aguarde..."
        )

        self.consensus_label.config(
            text="As duas IAs estão analisando a imagem..."
        )

        self.score_label.config(text="")
        self.status_label.config(text="Processamento em andamento...")

        threading.Thread(
            target=self.executar_analise,
            args=(path,),
            daemon=True
        ).start()

    def executar_analise(self, path):

        try:

            imagem = self.preparar_imagem(path)

            resultado_ia = self.analisar_mobilenet(
                imagem
            )

            resultado_groq = self.analisar_groq(
                path
            )

            comparacao = self.comparar_resultados(
                resultado_ia,
                resultado_groq
            )

            self.root.after(
                0,
                lambda: self.exibir_resultados(
                    path,
                    resultado_ia,
                    resultado_groq,
                    comparacao
                )
            )

        except Exception as erro:

            self.root.after(
                0,
                lambda mensagem=str(erro): self.exibir_erro(mensagem)
            )



    def exibir_resultados(
        self,
        path,
        resultado_ia,
        resultado_groq,
        comparacao
    ):

        nome_ia = traduzir_fruta(
            resultado_ia["classe"]
        )

        nome_groq = traduzir_fruta(
            resultado_groq["classe"]
        )

        confianca_ia = resultado_ia["confianca"] * 100
        confianca_groq = resultado_groq["confianca"] * 100

        self.mobile_result.config(
            text=nome_ia
        )

        self.mobile_confidence.config(
            text=f"Confiança estimada: {confianca_ia:.1f}%"
        )

        self.groq_result.config(
            text=nome_groq
        )

        self.groq_confidence.config(
            text=f"Confiança declarada: {confianca_groq:.1f}%"
        )

        preview = Image.open(path).convert("RGB")
        preview.thumbnail((500, 500), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(preview)

        self.image_area.config(
            image=self.photo,
            text=""
        )

        if comparacao["concordaram"]:

            self.consensus_label.config(
                text=f"AS DUAS IAs CONCORDAM: {nome_ia}",
                fg="#15803D"
            )

        elif comparacao["empate"]:

            self.consensus_label.config(
                text="EMPATE: as duas avaliações têm pontuação igual.",
                fg="#B45309"
            )

        else:

            nome_vencedora = traduzir_fruta(
                comparacao["vencedora"]
            )

            self.consensus_label.config(
                text=f"MAIOR PONTUAÇÃO: {nome_vencedora}",
                fg="#2563EB"
            )

        pontuacoes = comparacao["pontuacoes"]

        textos = [
            f"{traduzir_fruta(classe)}: {valor:.1f}%"
            for classe, valor in pontuacoes.items()
        ]

        self.score_label.config(
            text="Pontuação relativa: " + " | ".join(textos)
        )

        if comparacao["concordaram"]:
            diferenca = 0
        else:
            diferenca = (
                pontuacoes.get(resultado_ia["classe"], 0)
                - pontuacoes.get(resultado_groq["classe"], 0)
            )

        self.desenhar_balanca(diferenca)

        self.status_label.config(
            text=(
                f"Análise concluída | "
                f"{len(self.classes)} classes permitidas | "
                f"Modelo: {self.model_path.name}"
            )
        )

        self.analisando = False

    def exibir_erro(self, mensagem):

        self.analisando = False

        self.mobile_result.config(
            text="Falha na análise",
            fg="#B91C1C"
        )

        self.groq_result.config(
            text="Falha na análise",
            fg="#B91C1C"
        )

        self.consensus_label.config(
            text="Não foi possível comparar as duas avaliações.",
            fg="#B91C1C"
        )

        self.status_label.config(
            text="Erro durante o processamento."
        )

        messagebox.showerror(
            "Erro no Frutix",
            mensagem
        )


def main():

    root = tk.Tk()
    root.title("iconeap")
    root.iconbitmap(
        str(Path(__file__).parent / "iconeap.ico")
    )

    try:
        app = FruitApp(root)

    except Exception as erro:

        messagebox.showerror(
            "Erro ao iniciar",
            str(erro)
        )

        root.destroy()
        return

    root.mainloop()


if __name__ == "__main__":
    main()

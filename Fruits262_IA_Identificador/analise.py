
import pandas as pd
from pathlib import Path

pasta = Path("resultados")

arquivo_relatorio = pasta / "relatorio_classes.csv"
arquivo_matriz = pasta / "matriz_confusao.csv"
relatorio = pd.read_csv(arquivo_relatorio)

print("\n========== RESUMO GERAL ==========")

total_imagens = relatorio["Imagens_validacao"].sum()
total_acertos = relatorio["Acertos"].sum()
acuracia = total_acertos / total_imagens * 100

print(f"Imagens avaliadas: {total_imagens}")
print(f"Acertos: {total_acertos}")
print(f"Erros: {total_imagens - total_acertos}")
print(f"Acurácia geral: {acuracia:.2f}%")

print("\n========== DESEMPENHO DAS CLASSES ==========")

print("\nClasses com 0% de acerto:")
print(
    relatorio[relatorio["Acuracia"] == 0]
    [["Fruta", "Acuracia"]]
    .to_string(index=False)
)

print("\nClasses com 100% de acerto:")
print(
    relatorio[relatorio["Acuracia"] == 100]
    [["Fruta", "Acuracia"]]
    .to_string(index=False)
)

# Lê a matriz de confusão
matriz = pd.read_csv(arquivo_matriz, index_col=0)
matriz = matriz.apply(pd.to_numeric, errors="coerce").fillna(0)

print("\n========== MAIORES CONFUSÕES ==========")

confusoes = []

for fruta_real in matriz.index:
    for fruta_prevista in matriz.columns:
        quantidade = matriz.loc[fruta_real, fruta_prevista]

        if fruta_real != fruta_prevista and quantidade > 0:
            confusoes.append(
                (fruta_real, fruta_prevista, int(quantidade))
            )

confusoes.sort(key=lambda x: x[2], reverse=True)

for real, prevista, quantidade in confusoes[:30]:
    print(f"{real} foi confundida com {prevista}: {quantidade} vez(es)")

print("\nAnálise concluída!")

import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import (
    SimpleDocTemplate,
    PageBreak,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

st.set_page_config(
    page_title="IA de Soluções Técnicas",
    layout="centered"
)

st.title("🔧 IA de Soluções Técnicas")

st.write(
    "Sistema inteligente de recomendação de soluções técnicas."
)
nome_tecnico = st.text_input(
    "Nome do Técnico"
)
DATASET_PATH = "dataset/dataset-limpo3.csv"
FEEDBACK_PATH = "feedbacks.csv"

if not os.path.exists(FEEDBACK_PATH):
    pd.DataFrame(columns=[
        "tecnico",
        "familia",
        "categoria",
        "defeito",
        "solucao",
        "feedback"
    ]).to_csv(
        FEEDBACK_PATH,
        index=False
    )

def carregar_feedbacks():
    return pd.read_csv(
        FEEDBACK_PATH
    )

@st.cache_data
def carregar_dataset():
    df = pd.read_csv(DATASET_PATH)
    df = df[
        [
            "familia_descricao",
            "categoria_defeito",
            "defeito_constatado_descricao",
            "solucao_descricao",
            "categoria_solucao"
        ]
    ]
    df = df.dropna()
    df = df.drop_duplicates()


    df["texto_base"] = (
        "Família: "
        + df["familia_descricao"].astype(str)

        + " Categoria: "
        + df["categoria_defeito"].astype(str)

        + " Defeito: "
        + df["defeito_constatado_descricao"].astype(str)
    )
    return df

df = carregar_dataset()

def gerar_relatorio_pdf(
    familia,
    categoria,
    defeito,
    resultados,
    nome_tecnico
):

    feedbacks = carregar_feedbacks()
    meus_feedbacks = feedbacks[
        feedbacks["tecnico"] == nome_tecnico
    ]
    total = len(meus_feedbacks)
    positivos = len(
        meus_feedbacks[
            meus_feedbacks["feedback"] == "positivo"
        ]
    )
    negativos = len(
        meus_feedbacks[
            meus_feedbacks["feedback"] == "negativo"
        ]
    )
    if total > 0:
        taxa_aprovacao = (
            positivos / total
        ) * 100

        precisao = (
            positivos /
            (positivos + negativos)
        ) if (positivos + negativos) > 0 else 0

        recall = (
            positivos / total
        ) if total > 0 else 0
    else:
        taxa_aprovacao = 0
        precisao = 0
        recall = 0

    similaridade_media = (
        resultados["similaridade"]
        .mean()
    )

    nome_arquivo = (
        f"relatorio_{nome_tecnico}.pdf"
    )

    doc = SimpleDocTemplate(
        nome_arquivo
    )

    estilos = getSampleStyleSheet()

    elementos = []

    elementos.append(
        Paragraph(
            "RELATÓRIO TÉCNICO DE ATENDIMENTO",
            estilos["Title"]
        )
    )

    elementos.append(
        Spacer(1, 20)
    )

    elementos.append(
        Paragraph(
            f"<b>Técnico:</b> {nome_tecnico}",
            estilos["Normal"]
        )
    )

    elementos.append(
        Paragraph(
            f"<b>Data:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            estilos["Normal"]
        )
    )

    elementos.append(
        Spacer(1, 15)
    )

    elementos.append(
        Paragraph(
            f"<b>Família:</b> {familia}",
            estilos["Normal"]
        )
    )

    elementos.append(
        Paragraph(
            f"<b>Categoria:</b> {categoria}",
            estilos["Normal"]
        )
    )

    elementos.append(
        Paragraph(
            f"<b>Defeito:</b> {defeito}",
            estilos["Normal"]
        )
    )

    elementos.append(
        Spacer(1, 20)
    )

    elementos.append(
        Paragraph(
            "Soluções Recomendadas",
            estilos["Heading2"]
        )
    )

    for i, row in enumerate(
        resultados.itertuples(),
        start=1
    ):

        elementos.append(
            Paragraph(
                f"<b>Solução {i}</b>",
                estilos["Heading3"]
            )
        )

        elementos.append(
            Paragraph(
                row.solucao_descricao,
                estilos["Normal"]
            )
        )

        elementos.append(
            Paragraph(
                f"Categoria: {row.categoria_solucao}",
                estilos["Normal"]
            )
        )

        elementos.append(
            Paragraph(
                f"Similaridade: {row.similaridade:.2f}",
                estilos["Normal"]
            )
        )

        elementos.append(
            Spacer(1, 10)
        )

    elementos.append(
        PageBreak()
    )

    elementos.append(
        Paragraph(
            "Indicadores de Desempenho",
            estilos["Heading1"]
        )
    )

    elementos.append(
        Spacer(1, 10)
    )

    elementos.append(
        Paragraph(
            f"Total de avaliações: {total}",
            estilos["Normal"]
        )
    )

    elementos.append(
        Paragraph(
            f"Feedbacks positivos: {positivos}",
            estilos["Normal"]
        )
    )

    elementos.append(
        Paragraph(
            f"Feedbacks negativos: {negativos}",
            estilos["Normal"]
        )
    )

    elementos.append(
        Paragraph(
            f"Taxa de aprovação: {taxa_aprovacao:.2f}%",
            estilos["Normal"]
        )
    )

    elementos.append(
        Paragraph(
            f"Precisão (Precision): {precisao:.2f}",
            estilos["Normal"]
        )
    )

    elementos.append(
        Paragraph(
            f"Recall: {recall:.2f}",
            estilos["Normal"]
        )
    )

    elementos.append(
        Paragraph(
            f"Similaridade Média: {similaridade_media:.2f}",
            estilos["Normal"]
        )
    )

    doc.build(elementos)

    return nome_arquivo

@st.cache_resource
def carregar_modelo():

    modelo = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    embeddings = modelo.encode(
        df["texto_base"].tolist()
    )

    return modelo, embeddings

modelo, embeddings_dataset = carregar_modelo()

familias = sorted(
    df["familia_descricao"].unique()
)

categorias = sorted(
    df["categoria_defeito"].unique()
)

familia = st.selectbox(
    "Família do Produto",
    familias
)

categoria = st.selectbox(
    "Categoria do Defeito",
    categorias
)

defeito = st.text_area(
    "Descreva o defeito"
)

def recomendar_solucoes(
    familia,
    categoria,
    defeito,
    top_k=3
):

    consulta = (
        f"Família: {familia} "
        f"Categoria: {categoria} "
        f"Defeito: {defeito}"

    )

    embedding = modelo.encode(
        [consulta]
    )

    similaridades = cosine_similarity(
        embedding,
        embeddings_dataset
    )[0]

    df["similaridade"] = similaridades

    resultados = (
        df.sort_values(
            "similaridade",
            ascending=False
        )

        .drop_duplicates(
            subset=["solucao_descricao"]
        )

        .head(top_k)

    )

    return resultados

if "resultados" not in st.session_state:
    st.session_state.resultados = None

if "ultima_busca" not in st.session_state:
    st.session_state.ultima_busca = None

if st.button("Buscar Soluções"):
    resultados = recomendar_solucoes(
        familia,
        categoria,
        defeito
    )

    st.session_state.resultados = resultados

    st.session_state.ultima_busca = {
        "familia": familia,
        "categoria": categoria,
        "defeito": defeito
    }

if st.session_state.resultados is not None:
    resultados = st.session_state.resultados
    busca = st.session_state.ultima_busca

    st.subheader("Top 3 Soluções")

    for i, row in enumerate(
        resultados.itertuples(),
        start=1
    ):

        st.markdown(
            f"## 🔹 Solução {i}"
        )

        st.write(
            row.solucao_descricao
        )

        st.write(
            f"Similaridade: {row.similaridade:.2f}"
        )

        feedbacks = carregar_feedbacks()
        feedback_existente = feedbacks[
    (
        feedbacks["defeito"]
        == busca["defeito"]
    )
    &
    (
        feedbacks["solucao"]
        == row.solucao_descricao
    )
]

        positivos = len(
            feedback_existente[
                feedback_existente["feedback"]
                == "positivo"
            ]
        )

        negativos = len(
            feedback_existente[
                feedback_existente["feedback"]
                == "negativo"
            ]
        )

        st.write(
            f"👍 {positivos} | 👎 {negativos}"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(
                f"👍 Ajudou {i}",
                key=f"positivo_{i}"
            ):

                pd.DataFrame([{
                    "tecnico": nome_tecnico,
                    "familia": busca["familia"],
                    "categoria": busca["categoria"],
                    "defeito": busca["defeito"],
                    "solucao": row.solucao_descricao,
                    "feedback": "positivo"
                }]).to_csv(
                    FEEDBACK_PATH,
                    mode="a",
                    header=False,
                    index=False
                )

                st.rerun()

        with col2:
            if st.button(
                f"👎 Não ajudou {i}",
                key=f"negativo_{i}"
            ):

                pd.DataFrame([{
                    "tecnico": nome_tecnico,
                    "familia": busca["familia"],
                    "categoria": busca["categoria"],
                    "defeito": busca["defeito"],
                    "solucao": row.solucao_descricao,
                    "feedback": "negativo"
                }]).to_csv(
                    FEEDBACK_PATH,
                    mode="a",
                    header=False,
                    index=False
                )

                st.rerun()

        with col3:
            if st.button(
                f"❌ Desfazer {i}",
                key=f"desfazer_{i}"
            ):

                filtro = (
    (feedbacks["tecnico"] == nome_tecnico)
    &
    (feedbacks["defeito"] == busca["defeito"])
    &
    (feedbacks["solucao"] == row.solucao_descricao)
)

                indices = feedbacks[
                    filtro
                ].index

                if len(indices) > 0:

                    feedbacks = feedbacks.drop(
                        indices[-1]
                    )

                    feedbacks.to_csv(
                        FEEDBACK_PATH,
                        index=False
                    )

                    st.rerun()

        st.divider()

if st.session_state.resultados is not None:
    if st.button(
        "📄 Gerar Relatório PDF"
    ):

        arquivo_pdf = gerar_relatorio_pdf(
            busca["familia"],
            busca["categoria"],
            busca["defeito"],
            resultados,
            nome_tecnico
        )

        with open(
            arquivo_pdf,
            "rb"
        ) as pdf:

            st.download_button(
                label="⬇️ Baixar Relatório",
                data=pdf,
                file_name=arquivo_pdf,
                mime="application/pdf"
            )
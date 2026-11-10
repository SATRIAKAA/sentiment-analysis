from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.sentiment_analysis import MODEL_NAMES, train_and_evaluate


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "dataset.csv"


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes | None = None) -> pd.DataFrame:
    if file_bytes is None:
        return pd.read_csv(DEFAULT_DATASET)
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


@st.cache_resource(show_spinner="Melatih dan mengevaluasi model...")
def cached_train_with_attributes(
    dataset: pd.DataFrame,
    test_size: float,
    text_column: str,
    label_column: str,
):
    return train_and_evaluate(
        dataset,
        test_size=test_size,
        text_column=text_column,
        label_column=label_column,
    )


st.set_page_config(
    page_title="Analisis Sentimen Ulasan",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17233c;
        --muted: #74819a;
        --purple: #6658d3;
        --line: #e8ebf3;
        --surface: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'Aptos', 'Segoe UI', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background:
            radial-gradient(ellipse at 2% 0%, rgba(108, 91, 214, .08), transparent 34%),
            #f5f6fa;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], footer { visibility: hidden; }
    .block-container {
        max-width: 1440px;
        padding: 2rem 3rem 3.5rem;
    }
    h1, h2, h3 {
        font-family: 'Aptos Display', 'Segoe UI', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -.035em;
    }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.05rem !important; }
    [data-testid="stSidebar"] {
        background: #20233c;
    }
    [data-testid="stSidebar"] * { color: #f3f2ff; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,.07);
        border: 1px dashed rgba(255,255,255,.28);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: #f3f2ff;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div {
        background: rgba(255,255,255,.09);
        border-color: rgba(255,255,255,.17);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #b7bdd3;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.25rem 2.5rem;
        margin: .2rem 0 1.6rem;
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 24px;
        color: white;
        background: linear-gradient(118deg, #242744 0%, #353564 56%, #6658d3 120%);
        box-shadow: 0 16px 38px rgba(37, 39, 72, .15);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 290px;
        height: 290px;
        right: -72px;
        top: -150px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 0 0 35px rgba(255,255,255,.035), 0 0 0 70px rgba(255,255,255,.025);
    }
    .hero-kicker {
        margin-bottom: .65rem;
        color: #c9c4ff;
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0 0 .45rem;
        color: #fff !important;
        font-size: clamp(1.8rem, 3.2vw, 2.7rem) !important;
        letter-spacing: -.055em;
    }
    .hero p {
        max-width: 680px;
        margin: 0;
        color: #d3d5e7;
        font-size: 1rem;
        line-height: 1.65;
    }
    .hero-tags { display: flex; gap: .55rem; margin-top: 1.2rem; flex-wrap: wrap; }
    .hero-tags span {
        padding: .38rem .75rem;
        border: 1px solid rgba(255,255,255,.17);
        border-radius: 99px;
        background: rgba(255,255,255,.08);
        color: #f3f2ff;
        font-size: .78rem;
        font-weight: 600;
    }
    [data-testid="stMetric"] {
        padding: 1rem 1.15rem;
        border: 1px solid var(--line);
        border-radius: 15px;
        background: var(--surface);
        box-shadow: 0 5px 16px rgba(34, 43, 74, .035);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: .79rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Aptos Display', 'Segoe UI', sans-serif;
        font-weight: 800;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 18px;
        background: rgba(255,255,255,.92);
        box-shadow: 0 6px 20px rgba(34, 43, 74, .035);
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] h3 {
        margin-top: .2rem;
    }
    .st-key-prediction_panel [data-testid="stMarkdownContainer"],
    .st-key-prediction_panel [data-testid="stMarkdownContainer"] h4,
    .st-key-prediction_panel [data-testid="stMarkdownContainer"] a,
    .st-key-prediction_panel [data-testid="stCaptionContainer"],
    .st-key-prediction_panel [data-testid="stCaptionContainer"] p {
        color: #000000 !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stButton"] button {
        min-height: 2.8rem;
        border: 0;
        border-radius: 10px;
        background: linear-gradient(110deg, #6658d3, #8171e8);
        box-shadow: 0 7px 16px rgba(102,88,211,.2);
        font-weight: 700;
        transition: transform .18s ease, box-shadow .18s ease;
    }
    [data-testid="stButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 20px rgba(102,88,211,.27);
    }
    [data-testid="stTextArea"] textarea {
        border-radius: 12px;
        border-color: var(--line);
        background: #fbfbfe;
        color: #000000;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #8d81e5;
        box-shadow: 0 0 0 1px #8d81e5;
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    @media (max-width: 800px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        .hero { padding: 1.5rem; border-radius: 18px; }
        .hero:after { right: -190px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Sentiment intelligence · Dashboard</div>
        <h1>Pahami suara pelanggan Anda.</h1>
        <p>Analisis ulasan berbahasa Indonesia, bandingkan performa model,
        dan temukan sentimen di balik setiap kata.</p>
        <div class="hero-tags">
            <span>Support Vector Machine</span>
            <span>Naive Bayes</span>
            <span>Linear Regression</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    st.caption("Siapkan data dan atur eksperimen model.")
    uploaded_file = st.file_uploader(
        "Unggah dataset CSV",
        type=["csv"],
    )
    test_size = st.slider(
        "Proporsi data pengujian",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        format="%.0f%%",
    )
    test_size /= 100
    selected_model = st.selectbox("Model untuk prediksi", MODEL_NAMES)
    st.divider()

if uploaded_file is None and not DEFAULT_DATASET.is_file():
    st.info(
        "Dataset contoh tidak disertakan di repository. "
        "Unggah dataset CSV pada sidebar untuk mulai menganalisis."
    )
    st.stop()

try:
    file_bytes = uploaded_file.getvalue() if uploaded_file else None
    dataset = load_dataset(file_bytes)
except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
    st.error(f"Dataset tidak dapat dibaca: {error}")
    st.stop()

if dataset.shape[1] < 2:
    st.error("Dataset harus memiliki minimal dua kolom: teks ulasan dan label sentimen.")
    st.stop()

column_names = list(dataset.columns)
text_default = next(
    (
        index
        for index, name in enumerate(column_names)
        if str(name).strip().casefold()
        in {"reviews", "review", "text", "ulasan", "komentar", "content"}
    ),
    0,
)
label_default = next(
    (
        index
        for index, name in enumerate(column_names)
        if str(name).strip().casefold()
        in {"label", "sentiment", "sentimen", "target", "class"}
        and index != text_default
    ),
    1 if text_default != 1 else 0,
)

with st.sidebar:
    st.markdown("**Pilih atribut dataset**")
    text_column = st.selectbox(
        "Kolom teks ulasan",
        column_names,
        index=text_default,
        key="text_attribute",
    )
    label_column = st.selectbox(
        "Kolom label sentimen",
        column_names,
        index=label_default,
        key="label_attribute",
    )
    st.caption(
        "Dapat memakai nama kolom apa pun. Label harus 2 atau 3 kelas: "
        "negatif/netral/positif, atau angka berurutan."
    )
    st.caption("Pembagian data menggunakan stratifikasi dan random state 42.")

if text_column == label_column:
    st.error("Pilih kolom berbeda untuk teks ulasan dan label sentimen.")
    st.stop()

try:
    results = cached_train_with_attributes(
        dataset,
        test_size,
        text_column,
        label_column,
    )
except (OSError, ValueError) as error:
    st.error(f"Dataset tidak dapat diproses: {error}")
    st.stop()

clean_data = results["clean_data"]
actual_labels = set(clean_data["label"].dropna().unique())
sentiment_classes = [
    label for label in ("Negatif", "Netral", "Positif") if label in actual_labels
]
if not sentiment_classes:
    st.error("Label sentimen tidak ditemukan setelah dataset dibersihkan.")
    st.stop()
st.markdown("## 📌 Ringkasan dataset")
st.caption(f"Atribut teks: `{text_column}` · Atribut sentimen: `{label_column}`")
summary_left, summary_middle, summary_right = st.columns(3)
with summary_left:
    st.metric("Jumlah ulasan", f"{len(clean_data):,}", help="Ulasan setelah validasi data.")
with summary_middle:
    st.metric(
        "Kelas sentimen",
        len(sentiment_classes),
        help=", ".join(sentiment_classes),
    )
with summary_right:
    st.metric(
        "Fitur TF-IDF",
        f"{results['feature_count']:,}",
        help="Kata dan pasangan kata yang dipelajari.",
    )

distribution_column, performance_column = st.columns(2)
with distribution_column:
    with st.container(border=True):
        st.markdown("### Distribusi sentimen")
        st.caption("Sebaran label dalam dataset yang digunakan.")
        counts = (
            clean_data["label"]
            .value_counts()
            .reindex(sentiment_classes, fill_value=0)
            .astype(int)
        )
        figure, axis = plt.subplots(figsize=(6, 3.4))
        figure.patch.set_facecolor("#ffffff")
        axis.set_facecolor("#ffffff")
        sentiment_colors = {
            "Negatif": "#f08b8b",
            "Netral": "#e9b957",
            "Positif": "#56b894",
        }
        bars = axis.bar(
            counts.index,
            counts.values,
            width=0.52,
            color=[sentiment_colors[label] for label in counts.index],
            zorder=3,
        )
        axis.set_ylabel("Jumlah ulasan", color="#74819a")
        max_count = int(counts.max()) if not counts.empty else 0
        axis.set_ylim(0, max(max_count * 1.2, 1))
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#e8ebf3")
        axis.tick_params(axis="both", colors="#74819a", length=0, pad=8)
        axis.grid(axis="y", color="#edf0f6", linewidth=.8, zorder=0)
        for bar, count in zip(bars, counts.values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                count,
                f"{count:,}",
                ha="center",
                va="bottom",
                color="#17233c",
                fontweight="bold",
            )
        figure.tight_layout()
        st.pyplot(figure, clear_figure=True)

with performance_column:
    with st.container(border=True):
        st.markdown("### Performa model")
        st.caption(f"Evaluasi pada data uji · {test_size:.0%} dari dataset")
        metrics = pd.DataFrame(results["metrics"]).T
        metrics.index.name = "Model"
        st.dataframe(
            metrics.style.format(
                {
                    "Akurasi": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1-score": "{:.3f}",
                }
            ),
            width="stretch",
        )

st.markdown("## 🎯 Evaluasi model")
st.caption("Confusion matrix menunjukkan jumlah prediksi benar dan keliru untuk setiap kelas.")
matrix_columns = st.columns(len(MODEL_NAMES))
for column, model_name in zip(matrix_columns, MODEL_NAMES):
    with column:
        with st.container(border=True):
            st.markdown(f"### {model_name}")
            figure, axis = plt.subplots(figsize=(4, 3.2))
            figure.patch.set_facecolor("#ffffff")
            sns.heatmap(
                results["confusion_matrices"][model_name],
                annot=True,
                fmt="d",
                cmap=sns.light_palette("#6658d3", as_cmap=True),
                cbar=False,
                linewidths=2,
                linecolor="#ffffff",
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes,
                annot_kws={"fontsize": 13, "fontweight": "bold"},
                ax=axis,
            )
            axis.set_facecolor("#ffffff")
            axis.set_xlabel("Prediksi", color="#74819a", labelpad=9)
            axis.set_ylabel("Aktual", color="#74819a", labelpad=9)
            axis.tick_params(colors="#59657d", length=0)
            figure.tight_layout()
            st.pyplot(figure, clear_figure=True)

st.markdown("## ✨ Coba prediksi sentimen")
st.caption("Masukkan satu ulasan untuk melihat hasil prediksi dari model pilihan.")
with st.container(border=True, key="prediction_panel"):
    review_column, prediction_column = st.columns([3, 1])
    with review_column:
        review = st.text_area(
            "Tulis ulasan yang ingin dianalisis",
            placeholder="Contoh: Bahannya bagus, lembut, dan nyaman dipakai.",
            height=130,
            label_visibility="visible",
        )
    with prediction_column:
        st.markdown(f"**Model aktif**")
        st.markdown(f"#### {selected_model}")
        st.caption("Model dilatih menggunakan dataset saat ini.")
        predict_clicked = st.button("Prediksi sentimen →", type="primary")
    if predict_clicked:
        if not review.strip():
            st.warning("Masukkan ulasan terlebih dahulu.")
        else:
            prediction = results["models"][selected_model].predict([review])[0]
            if prediction == "Positif":
                st.success(f"Hasil analisis: **{prediction.upper()}** · Ulasan menunjukkan sentimen positif.")
            elif prediction == "Netral":
                st.info(f"Hasil analisis: **{prediction.upper()}** · Ulasan menunjukkan sentimen netral.")
            else:
                st.error(f"Hasil analisis: **{prediction.upper()}** · Ulasan menunjukkan sentimen negatif.")

with st.expander("Lihat sampel dataset yang digunakan"):
    st.dataframe(clean_data.head(20), width="stretch")

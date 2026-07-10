"""
Dashboard Deteksi Fraud Keuangan & Analisis Jaringan Money Laundering
Kelompok 2 - Data Mining dan Analisis Jaringan Graf
Universitas Sebelas April

Mereplikasi alur CRISP-DM dari notebook Google Colab:
Business Understanding -> Data Understanding -> Data Preparation ->
Modeling (Decision Tree) -> Evaluation -> Graph Analytics -> Kesimpulan
"""

import io
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

# =============================================================
# KONFIGURASI HALAMAN & STYLE
# =============================================================
st.set_page_config(
    page_title="Deteksi Fraud Keuangan | Kelompok 2",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {"normal": "#3498db", "fraud": "#e74c3c"}
sns.set_style("whitegrid")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1rem;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #3498db;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = [
    "step",
    "type_encoded",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
TYPE_MAP = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}


# =============================================================
# DATA HELPERS
# =============================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_paysim(n_rows: int = 60_000, fraud_ratio: float = 0.0013, seed: int = 42) -> pd.DataFrame:
    """Membuat data sintetis dengan struktur mirip PaySim, untuk demo tanpa dataset asli."""
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_rows * fraud_ratio), 30)
    n_normal = n_rows - n_fraud

    types_normal = rng.choice(
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        size=n_normal,
        p=[0.22, 0.35, 0.06, 0.34, 0.03],
    )
    types_fraud = rng.choice(["CASH_OUT", "TRANSFER"], size=n_fraud, p=[0.5, 0.5])

    def make_block(n, types, fraud_flag):
        step = rng.integers(1, 744, size=n)
        amount = np.round(rng.exponential(scale=90_000 if fraud_flag else 25_000, size=n) + 1, 2)
        old_orig = np.round(rng.exponential(scale=60_000, size=n), 2)
        if fraud_flag:
            # pola khas notebook: saldo pengirim jadi 0 setelah transaksi
            new_orig = np.where(rng.random(n) < 0.7, 0.0, np.maximum(old_orig - amount, 0))
        else:
            new_orig = np.maximum(old_orig - amount, 0)
        old_dest = np.round(rng.exponential(scale=40_000, size=n), 2)
        if fraud_flag:
            zero_dest = rng.random(n) < 0.5
            new_dest = np.where(zero_dest, 0.0, old_dest + amount)
            old_dest = np.where(zero_dest, 0.0, old_dest)
        else:
            new_dest = old_dest + amount

        name_orig = np.array([f"C{rng.integers(10**8, 10**9)}" for _ in range(n)])
        name_dest = np.array([f"C{rng.integers(10**8, 10**9)}" for _ in range(n)])

        return pd.DataFrame(
            {
                "step": step,
                "type": types,
                "amount": amount,
                "nameOrig": name_orig,
                "oldbalanceOrg": old_orig,
                "newbalanceOrig": new_orig,
                "nameDest": name_dest,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
                "isFraud": fraud_flag,
                "isFlaggedFraud": 0,
            }
        )

    df_normal = make_block(n_normal, types_normal, 0)
    df_fraud = make_block(n_fraud, types_fraud, 1)

    # Sedikit akun fraud dipakai berulang supaya graph analytics punya pola
    reused_accounts = df_fraud["nameOrig"].sample(frac=0.4, random_state=seed).tolist()
    for i in range(len(df_fraud)):
        if rng.random() < 0.3 and reused_accounts:
            df_fraud.iat[i, df_fraud.columns.get_loc("nameDest")] = rng.choice(reused_accounts)

    df = pd.concat([df_normal, df_fraud], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["isFraud"] = df["isFraud"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def get_fig_download_button(fig, label, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="image/png")


# =============================================================
# SIDEBAR - KONTROL & UPLOAD DATA
# =============================================================
with st.sidebar:
    st.markdown("## ⚙️ Panel Kontrol")
    st.markdown("---")

    st.markdown("### 1️⃣ Sumber Data")
    data_source = st.radio(
        "Pilih sumber dataset PaySim",
        ["Upload dataset saya (.csv)", "Gunakan data sampel demo"],
        index=1,
    )

    uploaded_file = None
    demo_rows = 60_000
    if data_source == "Upload dataset saya (.csv)":
        uploaded_file = st.file_uploader(
            "Upload file PaySim (PS_20174392719_1491204439457_log.csv)",
            type=["csv"],
        )
        st.caption("Format harus sama dengan dataset PaySim dari Kaggle (ealaxi/paysim1).")
    else:
        demo_rows = st.slider("Jumlah baris data sampel", 10_000, 200_000, 60_000, step=10_000)
        st.caption("Data sintetis dibuat otomatis dengan pola menyerupai PaySim (untuk demo tanpa file asli).")

    st.markdown("---")
    st.markdown("### 2️⃣ Data Preparation")
    sampling_ratio = st.slider("Rasio under-sampling (normal : fraud)", 2, 20, 10)
    test_size = st.slider("Proporsi data testing", 0.1, 0.4, 0.2, step=0.05)

    st.markdown("---")
    st.markdown("### 3️⃣ Modeling")
    tuning_mode = st.radio("Mode hyperparameter", ["Grid search otomatis (seperti notebook)", "Atur manual"])
    manual_depth, manual_mss = 6, 2
    if tuning_mode == "Atur manual":
        manual_depth = st.slider("max_depth", 2, 15, 6)
        manual_mss = st.select_slider("min_samples_split", options=[2, 5, 10], value=2)

    st.markdown("---")
    st.markdown("### 4️⃣ Graph Analytics")
    min_txn_active = st.slider("Minimal transaksi akun aktif (untuk graf)", 1, 5, 1)

    st.markdown("---")
    run_button = st.button("🚀 Jalankan Analisis Lengkap", use_container_width=True, type="primary")

# =============================================================
# HEADER
# =============================================================
st.markdown('<p class="main-title">🕵️ Deteksi Fraud Keuangan & Analisis Jaringan Money Laundering</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Data Mining & Analisis Jaringan Graf — Decision Tree + Graph Analytics (NetworkX) | '
    "Dataset: PaySim (Kaggle)</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# =============================================================
# STATE INIT
# =============================================================
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False

# =============================================================
# JALANKAN PIPELINE SAAT TOMBOL DITEKAN
# =============================================================
if run_button:
    with st.spinner("Memuat data..."):
        if data_source == "Upload dataset saya (.csv)":
            if uploaded_file is None:
                st.error("Silakan upload file CSV dataset PaySim terlebih dahulu di sidebar.")
                st.stop()
            df = load_uploaded_csv(uploaded_file.getvalue())
        else:
            df = generate_synthetic_paysim(n_rows=demo_rows)

    required_cols = {
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        st.error(f"Dataset tidak sesuai format PaySim. Kolom hilang: {', '.join(missing_cols)}")
        st.stop()

    # ---------- DATA PREPARATION ----------
    with st.spinner("Melakukan under-sampling & encoding..."):
        df_fraud_all = df[df["isFraud"] == 1]
        n_normal_target = min(len(df_fraud_all) * sampling_ratio, len(df[df["isFraud"] == 0]))
        df_normal = df[df["isFraud"] == 0].sample(n=int(n_normal_target), random_state=42)
        df_sample = pd.concat([df_fraud_all, df_normal]).reset_index(drop=True)

        df_sample["type_encoded"] = df_sample["type"].map(TYPE_MAP)
        df_sample = df_sample.dropna(subset=["type_encoded"])
        df_sample["type_encoded"] = df_sample["type_encoded"].astype(int)

        X = df_sample[FEATURES]
        y = df_sample["isFraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

    # ---------- MODELING ----------
    tuning_results_df = None
    with st.spinner("Melatih model Decision Tree..."):
        if tuning_mode == "Grid search otomatis (seperti notebook)":
            max_depth_list = [3, 4, 5, 6, 7, 8, 10]
            min_samples_split_list = [2, 5, 10]
            results = []
            for depth in max_depth_list:
                for mss in min_samples_split_list:
                    temp_model = DecisionTreeClassifier(
                        max_depth=depth, min_samples_split=mss, random_state=42, class_weight="balanced"
                    )
                    temp_model.fit(X_train, y_train)
                    temp_pred = temp_model.predict(X_test)
                    f1_fraud = f1_score(y_test, temp_pred, pos_label=1, zero_division=0)
                    results.append({"max_depth": depth, "min_samples_split": mss, "f1_fraud": f1_fraud})
            tuning_results_df = pd.DataFrame(results).sort_values("f1_fraud", ascending=False).reset_index(drop=True)
            best_params = tuning_results_df.iloc[0]
            best_depth, best_mss = int(best_params["max_depth"]), int(best_params["min_samples_split"])
        else:
            best_depth, best_mss = manual_depth, manual_mss

        model_dt = DecisionTreeClassifier(
            max_depth=best_depth, min_samples_split=best_mss, random_state=42, class_weight="balanced"
        )
        model_dt.fit(X_train, y_train)

    # ---------- EVALUATION ----------
    y_pred = model_dt.predict(X_test)
    y_pred_proba = model_dt.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = float("nan")
    report_dict = classification_report(y_test, y_pred, target_names=["Normal", "Fraud"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # ---------- GRAPH ANALYTICS ----------
    with st.spinner("Membangun jaringan aliran dana & mendeteksi sindikat..."):
        df_fraud_only = df_sample[df_sample["isFraud"] == 1].copy()
        all_accounts = pd.concat([df_fraud_only["nameOrig"], df_fraud_only["nameDest"]])
        account_counts = all_accounts.value_counts()
        active_accounts = account_counts[account_counts > min_txn_active].index.tolist()

        df_graf = df_fraud_only[
            df_fraud_only["nameOrig"].isin(active_accounts) | df_fraud_only["nameDest"].isin(active_accounts)
        ]

        G = nx.DiGraph()
        for _, row in df_graf.iterrows():
            if G.has_edge(row["nameOrig"], row["nameDest"]):
                G[row["nameOrig"]][row["nameDest"]]["weight"] += row["amount"]
                G[row["nameOrig"]][row["nameDest"]]["count"] += 1
            else:
                G.add_edge(row["nameOrig"], row["nameDest"], weight=row["amount"], count=1)

        communities = []
        top_degree = pd.DataFrame(columns=["Akun", "Degree Centrality"])
        top_in = pd.DataFrame(columns=["Akun", "In-Degree"])
        top_out = pd.DataFrame(columns=["Akun", "Out-Degree"])
        degree_cent = {}

        if G.number_of_nodes() > 0:
            degree_cent = nx.degree_centrality(G)
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())

            top_degree = (
                pd.DataFrame(degree_cent.items(), columns=["Akun", "Degree Centrality"])
                .sort_values("Degree Centrality", ascending=False)
                .head(10)
            )
            top_in = (
                pd.DataFrame(in_degree.items(), columns=["Akun", "In-Degree"])
                .sort_values("In-Degree", ascending=False)
                .head(10)
            )
            top_out = (
                pd.DataFrame(out_degree.items(), columns=["Akun", "Out-Degree"])
                .sort_values("Out-Degree", ascending=False)
                .head(10)
            )
            if G.number_of_nodes() > 2:
                G_undirected = G.to_undirected()
                communities = sorted(nx.community.greedy_modularity_communities(G_undirected), key=len, reverse=True)

    # ---------- SIMPAN KE SESSION STATE ----------
    st.session_state.update(
        dict(
            pipeline_done=True,
            df=df,
            df_sample=df_sample,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            tuning_results_df=tuning_results_df,
            best_depth=best_depth,
            best_mss=best_mss,
            model_dt=model_dt,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            acc=acc,
            auc=auc,
            report_dict=report_dict,
            cm=cm,
            G=G,
            communities=communities,
            top_degree=top_degree,
            top_in=top_in,
            top_out=top_out,
            degree_cent=degree_cent,
            active_accounts=active_accounts,
            df_graf=df_graf,
        )
    )
    st.success("Analisis selesai! Jelajahi hasilnya lewat tab di bawah.")

# =============================================================
# TABS UTAMA
# =============================================================
tabs = st.tabs(
    [
        "📋 Business Understanding",
        "📊 Data Understanding",
        "🛠️ Data Preparation",
        "🌳 Modeling",
        "📈 Evaluation",
        "🕸️ Graph Analytics",
        "📝 Kesimpulan",
    ]
)

# ---------------- TAB 1: BUSINESS UNDERSTANDING ----------------
with tabs[0]:
    st.header("Business Understanding")
    st.markdown(
        """
        ### Latar Belakang
        Penipuan keuangan (*financial fraud*) merupakan ancaman serius bagi industri perbankan dan fintech.
        Setiap tahun, miliaran dolar kerugian terjadi akibat transaksi ilegal yang sulit dideteksi secara manual.

        ### Tujuan Bisnis
        - Mendeteksi transaksi fraud secara otomatis menggunakan *machine learning*
        - Memetakan jaringan aliran dana mencurigakan untuk mengidentifikasi sindikat *money laundering*
        - Memberikan insight kepada tim keamanan keuangan untuk tindakan preventif

        ### Pertanyaan Analitik
        1. Fitur transaksi apa yang paling berpengaruh dalam mendeteksi fraud?
        2. Akun mana yang menjadi pusat jaringan sindikat money laundering?
        3. Berapa akurasi model dalam mendeteksi transaksi fraud?

        ### Dataset
        - **Sumber:** PaySim — *Synthetic Financial Dataset for Fraud Detection* (Kaggle)
        - **Deskripsi:** Simulasi transaksi *mobile money* selama 30 hari
        - **Link:** https://www.kaggle.com/datasets/ealaxi/paysim1

        ### Metodologi: CRISP-DM
        1. Business Understanding
        2. Data Understanding
        3. Data Preparation
        4. Modeling
        5. Evaluation
        6. Graph Analytics & Deployment
        """
    )
    st.info("👈 Atur sumber data & parameter di sidebar, lalu klik **Jalankan Analisis Lengkap** untuk memulai.")

# ---------------- TAB 2: DATA UNDERSTANDING ----------------
with tabs[1]:
    st.header("Data Understanding")
    if not st.session_state.pipeline_done:
        st.warning("Jalankan analisis terlebih dahulu dari sidebar untuk melihat hasil EDA.")
    else:
        df = st.session_state.df
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Baris", f"{df.shape[0]:,}")
        c2.metric("Jumlah Kolom", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("5 Baris Pertama")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Distribusi Target (isFraud)")
        fraud_counts = df["isFraud"].value_counts()
        fraud_pct = df["isFraud"].value_counts(normalize=True) * 100
        colA, colB = st.columns(2)
        colA.metric("Normal (0)", f"{fraud_counts.get(0, 0):,}", f"{fraud_pct.get(0, 0):.2f}%")
        colB.metric("Fraud (1)", f"{fraud_counts.get(1, 0):,}", f"{fraud_pct.get(1, 0):.2f}%")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].pie(
            fraud_counts,
            labels=["Normal", "Fraud"],
            autopct="%1.2f%%",
            colors=[PALETTE["normal"], PALETTE["fraud"]],
            startangle=90,
        )
        axes[0].set_title("Distribusi Transaksi Normal vs Fraud")

        type_fraud = df.groupby(["type", "isFraud"]).size().unstack(fill_value=0)
        type_fraud.plot(kind="bar", ax=axes[1], color=[PALETTE["normal"], PALETTE["fraud"]])
        axes[1].set_title("Jumlah Transaksi per Tipe")
        axes[1].set_xlabel("Tipe Transaksi")
        axes[1].set_ylabel("Jumlah")
        axes[1].legend(["Normal", "Fraud"])
        axes[1].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Pola Saldo Tujuan pada Transaksi Fraud")
        fraud_only = df[df["isFraud"] == 1]
        if len(fraud_only) > 0:
            st.dataframe(fraud_only[["oldbalanceDest", "newbalanceDest"]].describe(), use_container_width=True)
            zero_dest = fraud_only[(fraud_only["oldbalanceDest"] == 0) & (fraud_only["newbalanceDest"] == 0)]
            pct_zero = len(zero_dest) / len(fraud_only) * 100
            st.markdown(
                f"**{len(zero_dest):,} dari {len(fraud_only):,} transaksi fraud ({pct_zero:.1f}%)** "
                "punya saldo tujuan 0 di awal dan akhir."
            )

# ---------------- TAB 3: DATA PREPARATION ----------------
with tabs[2]:
    st.header("Data Preparation")
    if not st.session_state.pipeline_done:
        st.warning("Jalankan analisis terlebih dahulu dari sidebar.")
    else:
        df_sample = st.session_state.df_sample
        c1, c2, c3 = st.columns(3)
        n_fraud = int((df_sample["isFraud"] == 1).sum())
        n_normal = int((df_sample["isFraud"] == 0).sum())
        c1.metric("Data Fraud", f"{n_fraud:,}")
        c2.metric("Data Normal (sampel)", f"{n_normal:,}")
        c3.metric("Rasio Fraud", f"{n_fraud / len(df_sample) * 100:.1f}%")

        st.subheader("Encoding Tipe Transaksi")
        st.dataframe(
            df_sample[["type", "type_encoded"]].drop_duplicates().sort_values("type_encoded"),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Fitur yang Digunakan")
        st.code(", ".join(FEATURES))

        st.subheader("Pembagian Data")
        X_train, X_test = st.session_state.X_train, st.session_state.X_test
        total = len(X_train) + len(X_test)
        c1, c2 = st.columns(2)
        c1.metric("Data Training", f"{len(X_train):,}", f"{len(X_train) / total * 100:.0f}%")
        c2.metric("Data Testing", f"{len(X_test):,}", f"{len(X_test) / total * 100:.0f}%")

# ---------------- TAB 4: MODELING ----------------
with tabs[3]:
    st.header("Modeling — Decision Tree Classifier")
    if not st.session_state.pipeline_done:
        st.warning("Jalankan analisis terlebih dahulu dari sidebar.")
    else:
        tuning_df = st.session_state.tuning_results_df
        if tuning_df is not None:
            st.subheader("Hasil Hyperparameter Tuning (Top 10)")
            st.dataframe(tuning_df.head(10), use_container_width=True, hide_index=True)

        st.success(
            f"Kombinasi terbaik: **max_depth={st.session_state.best_depth}**, "
            f"**min_samples_split={st.session_state.best_mss}**"
        )

        model_dt = st.session_state.model_dt
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Depth", model_dt.max_depth)
        c2.metric("Jumlah Node", model_dt.tree_.node_count)
        c3.metric("Jumlah Leaves", model_dt.get_n_leaves())

        st.subheader("Visualisasi Pohon Keputusan (3 Level Pertama)")
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(
            model_dt,
            feature_names=FEATURES,
            class_names=["Normal", "Fraud"],
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=9,
            ax=ax,
        )
        ax.set_title("Decision Tree — Deteksi Fraud Keuangan (3 Level Pertama)", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---------------- TAB 5: EVALUATION ----------------
with tabs[4]:
    st.header("Evaluation")
    if not st.session_state.pipeline_done:
        st.warning("Jalankan analisis terlebih dahulu dari sidebar.")
    else:
        acc, auc = st.session_state.acc, st.session_state.auc
        report_dict = st.session_state.report_dict
        c1, c2, c3 = st.columns(3)
        c1.metric("Akurasi", f"{acc * 100:.2f}%")
        c2.metric("ROC-AUC", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
        c3.metric("F1-Score Fraud", f"{report_dict['Fraud']['f1-score'] * 100:.1f}%")

        st.subheader("Classification Report")
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

        st.subheader("Visualisasi Evaluasi")
        model_dt = st.session_state.model_dt
        y_test, y_pred_proba = st.session_state.y_test, st.session_state.y_pred_proba
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.heatmap(
            st.session_state.cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"],
        )
        axes[0].set_title("Confusion Matrix")
        axes[0].set_ylabel("Aktual")
        axes[0].set_xlabel("Prediksi")

        if not np.isnan(auc):
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            axes[1].plot(fpr, tpr, color=PALETTE["fraud"], lw=2, label=f"AUC = {auc:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()

        importance_df = pd.DataFrame(
            {"Fitur": FEATURES, "Importance": model_dt.feature_importances_}
        ).sort_values("Importance", ascending=True)
        axes[2].barh(importance_df["Fitur"], importance_df["Importance"], color=PALETTE["normal"])
        axes[2].set_title("Feature Importance")
        axes[2].set_xlabel("Importance Score")

        plt.suptitle("Evaluasi Model Decision Tree — Deteksi Fraud", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Feature Importance (Tabel)")
        st.dataframe(
            importance_df.sort_values("Importance", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# ---------------- TAB 6: GRAPH ANALYTICS ----------------
with tabs[5]:
    st.header("Graph Analytics — Jaringan Aliran Dana")
    if not st.session_state.pipeline_done:
        st.warning("Jalankan analisis terlebih dahulu dari sidebar.")
    else:
        G = st.session_state.G
        c1, c2 = st.columns(2)
        c1.metric("Jumlah Node (Akun)", f"{G.number_of_nodes():,}")
        c2.metric("Jumlah Edge (Transaksi)", f"{G.number_of_edges():,}")

        if G.number_of_nodes() == 0:
            st.info("Tidak ada akun aktif (transaksi > ambang batas) untuk divisualisasikan pada data ini. Coba turunkan 'Minimal transaksi akun aktif' di sidebar.")
        else:
            colL, colR = st.columns(2)
            with colL:
                st.subheader("Top 10 Akun Paling Aktif")
                st.dataframe(st.session_state.top_degree, use_container_width=True, hide_index=True)
            with colR:
                st.subheader("Top 10 Penerima / Pengirim Dana")
                sub1, sub2 = st.tabs(["In-Degree", "Out-Degree"])
                with sub1:
                    st.dataframe(st.session_state.top_in, use_container_width=True, hide_index=True)
                with sub2:
                    st.dataframe(st.session_state.top_out, use_container_width=True, hide_index=True)

            st.subheader("Deteksi Komunitas (Sindikat)")
            communities = st.session_state.communities
            st.markdown(f"**Jumlah sindikat terdeteksi: {len(communities)}**")
            if communities:
                comm_rows = [
                    {"Sindikat": f"Sindikat {i + 1}", "Jumlah Akun": len(c), "Contoh Akun": ", ".join(list(c)[:3])}
                    for i, c in enumerate(communities[:5])
                ]
                st.dataframe(pd.DataFrame(comm_rows), use_container_width=True, hide_index=True)

            st.subheader("Visualisasi Jaringan")
            top_nodes = st.session_state.top_degree["Akun"].tolist()
            degree_cent = st.session_state.degree_cent
            node_colors = [PALETTE["fraud"] if n in top_nodes else "#85c1e9" for n in G.nodes()]
            node_sizes = [degree_cent[n] * 5000 + 200 for n in G.nodes()]

            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            pos1 = nx.spring_layout(G, seed=42, k=1)
            nx.draw_networkx_nodes(G, pos1, node_color=node_colors, node_size=node_sizes, alpha=0.85, ax=axes[0])
            nx.draw_networkx_edges(G, pos1, alpha=0.4, arrows=True, arrowsize=12, edge_color="gray", ax=axes[0])
            nx.draw_networkx_labels(G, pos1, labels={n: n[:8] for n in top_nodes}, font_size=7, ax=axes[0])
            axes[0].set_title("Jaringan Aliran Dana Fraud\n(Merah = Akun Kunci)", fontsize=12)
            axes[0].axis("off")

            degrees = [d for _, d in G.degree()]
            axes[1].hist(degrees, bins=15, color=PALETTE["normal"], edgecolor="white", alpha=0.8)
            axes[1].set_title("Distribusi Degree Node\n(Frekuensi Transaksi per Akun)", fontsize=12)
            axes[1].set_xlabel("Degree (Jumlah Koneksi)")
            axes[1].set_ylabel("Jumlah Akun")
            axes[1].axvline(np.mean(degrees), color="red", linestyle="--", label=f"Rata-rata: {np.mean(degrees):.2f}")
            axes[1].legend()

            plt.suptitle("Graph Analytics — Deteksi Sindikat Money Laundering", fontsize=14, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ---------------- TAB 7: KESIMPULAN ----------------
with tabs[6]:
    st.header("Kesimpulan & Rekomendasi")
    if not st.session_state.pipeline_done:
        st.warning("Jalankan analisis terlebih dahulu dari sidebar untuk melihat kesimpulan berbasis data aktual.")
    else:
        acc, auc = st.session_state.acc, st.session_state.auc
        report_dict = st.session_state.report_dict
        model_dt = st.session_state.model_dt
        communities = st.session_state.communities

        st.subheader("Ringkasan Hasil Model")
        summary_df = pd.DataFrame(
            {
                "Metrik": ["Akurasi", "ROC-AUC", "Recall Fraud", "Precision Fraud", "F1-Score Fraud"],
                "Nilai": [
                    f"{acc * 100:.2f}%",
                    f"{auc:.4f}" if not np.isnan(auc) else "N/A",
                    f"{report_dict['Fraud']['recall'] * 100:.1f}%",
                    f"{report_dict['Fraud']['precision'] * 100:.1f}%",
                    f"{report_dict['Fraud']['f1-score'] * 100:.1f}%",
                ],
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        importance_df = pd.DataFrame(
            {"Fitur": FEATURES, "Importance": model_dt.feature_importances_}
        ).sort_values("Importance", ascending=False)
        top_feat = importance_df.iloc[0]
        st.markdown(
            f"- Fitur paling penting: **{top_feat['Fitur']}** ({top_feat['Importance'] * 100:.1f}%)\n"
            f"- Model menggunakan `max_depth={st.session_state.best_depth}`, "
            f"`min_samples_split={st.session_state.best_mss}`, dan `class_weight='balanced'` "
            "untuk menangani data yang sangat *imbalanced*.\n"
            f"- Terdeteksi **{len(communities)} sindikat** melalui *community detection* (*greedy modularity*)."
        )

        st.markdown("---")
        st.subheader("📌 Rekomendasi")
        st.markdown(
            """
            1. **Terapkan model Decision Tree hasil tuning pada sistem monitoring transaksi real-time**,
               dengan prioritas pemeriksaan otomatis pada transaksi bertipe **CASH_OUT** dan **TRANSFER**.
            2. **Bangun aturan deteksi berbasis pola rantai (chain detection)**, bukan hanya deteksi hub sentral,
               karena sindikat cenderung beroperasi dalam kelompok kecil yang saling terpisah.
               Telusuri akun dengan in-degree ≥ 2 sebagai titik mule/perantara prioritas.
            3. **Tandai otomatis transaksi dengan pola "saldo tujuan nol sebelum dan sesudah"**
               sebagai sinyal tambahan verifikasi manual, dikombinasikan dengan probabilitas model
               (bukan keputusan biner semata) agar tim risk dapat memprioritaskan kasus dengan confidence tertinggi.
            """
        )

st.markdown("---")
st.caption("Dashboard dibuat berdasarkan notebook Kelompok 2 — Data Mining & Analisis Jaringan Graf | Universitas Sebelas April")

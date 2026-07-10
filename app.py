import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Detection - Colab", layout="wide")

st.title("🔍 Fraud Detection Dashboard")
st.markdown("**Synthetic Financial Datasets for Fraud Detection**")
st.caption("Berdasarkan Notebook Google Colab Kamu")

uploaded_file = st.file_uploader("Upload File Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, nrows=400000)  # Sampling agar tidak crash

    st.success(f"Dataset berhasil dimuat: {len(df):,} baris")

    # Filter
    st.sidebar.header("Filter")
    type_list = st.sidebar.multiselect("Tipe Transaksi", df['type'].unique(), default=[])
    fraud_status = st.sidebar.selectbox("Status Fraud", ["All", "Fraud", "Normal"])

    # Apply Filter
    filtered_df = df.copy()
    if type_list:
        filtered_df = filtered_df[filtered_df['type'].isin(type_list)]
    if fraud_status == "Fraud":
        filtered_df = filtered_df[filtered_df['isFraud'] == 1]
    elif fraud_status == "Normal":
        filtered_df = filtered_df[filtered_df['isFraud'] == 0]

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(filtered_df):,}")
    col2.metric("Fraud Cases", len(filtered_df[filtered_df['isFraud'] == 1]))
    rate = len(filtered_df[filtered_df['isFraud'] == 1]) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    col3.metric("Fraud Rate", f"{rate:.2f}%")
    col4.metric("Total Amount", f"Rp {filtered_df['amount'].sum():,.0f}")

    # Tabs seperti di Colab
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 EDA", "🔍 Fraud Analysis", "📉 Model Performance"])

    with tab1:
        st.subheader("Distribusi Tipe Transaksi")
        fig1 = px.bar(filtered_df['type'].value_counts().reset_index(), x='type', y='count', title="Transaction Type Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fraud by Type")
            fig2 = px.bar(filtered_df.groupby('type')['isFraud'].sum().reset_index(), x='type', y='isFraud')
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.subheader("Amount Distribution")
            fig3 = px.histogram(filtered_df, x='amount', color='isFraud', nbins=50)
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("Balance Analysis (Old vs New)")
        sample = filtered_df.sample(min(10000, len(filtered_df)))
        fig4 = px.scatter(sample, x='oldbalanceOrg', y='newbalanceOrig', color='isFraud', opacity=0.6)
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Fraud Rate per Step")
        if 'step' in filtered_df.columns:
            step_fraud = filtered_df.groupby('step')['isFraud'].mean().reset_index()
            fig5 = px.line(step_fraud, x='step', y='isFraud', title="Fraud Rate per Step")
            st.plotly_chart(fig5, use_container_width=True)

    with tab4:
        st.subheader("Model Performance Summary (dari Colab)")
        st.info("**Hasil Model di Colab kamu:**")
        st.write("- Akurasi tinggi pada kelas Fraud")
        st.write("- Fokus utama: CASH_OUT & TRANSFER")
        st.write("- Pola fraud: saldo tujuan sering 0")

        if 'isFraud' in filtered_df.columns:
            st.metric("Total Fraud Detected", len(filtered_df[filtered_df['isFraud']==1]))

    st.subheader("Data Preview")
    st.dataframe(filtered_df.head(50), use_container_width=True)

else:
    st.warning("Upload file dataset dari Colab kamu (PS_2017...log.csv)")
    st.info("Ukuran file besar? Gunakan sampling di atas.")
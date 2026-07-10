# Dashboard Deteksi Fraud Keuangan & Analisis Jaringan Money Laundering

Dashboard Streamlit ini mereplikasi seluruh alur notebook Colab **Kelompok 2**
(Data Mining & Analisis Jaringan Graf): Business Understanding → Data
Understanding → Data Preparation → Modeling (Decision Tree) → Evaluation →
Graph Analytics (NetworkX) → Kesimpulan.

## 🚀 Cara Menjalankan

1. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan dashboard:
   ```bash
   streamlit run app.py
   ```

3. Buka browser ke alamat yang muncul (biasanya `http://localhost:8501`).

## 📂 Sumber Data

Di sidebar tersedia dua opsi:

- **Upload dataset saya (.csv)** — upload file PaySim asli
  (`PS_20174392719_1491204439457_log.csv`) yang bisa diunduh dari
  [Kaggle: ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1).
  Pastikan kolomnya sesuai: `step, type, amount, nameOrig, oldbalanceOrg,
  newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud,
  isFlaggedFraud`.
- **Gunakan data sampel demo** — dashboard otomatis membuat data sintetis
  dengan pola menyerupai PaySim (imbalanced ~0.13% fraud, pola saldo tujuan
  nol, dsb) sehingga dashboard tetap bisa dicoba tanpa dataset asli.

> Catatan: dataset PaySim asli berukuran ±470MB (6,3 juta baris). Jika
> file terlalu besar untuk di-upload langsung ke Streamlit, pertimbangkan
> memotong sebagian baris terlebih dahulu (misalnya dengan `df.sample()`
> atau `head -n 500000` di terminal) sebelum upload.

## ⚙️ Parameter yang Bisa Diatur

- **Rasio under-sampling** — perbandingan data normal : fraud saat sampling
  (default 10x, sama seperti notebook).
- **Proporsi data testing** — default 20% (train/test split).
- **Mode hyperparameter** — grid search otomatis (sama seperti notebook:
  `max_depth` 3–10, `min_samples_split` 2/5/10, dievaluasi dengan F1-Score
  kelas Fraud) atau atur manual.
- **Minimal transaksi akun aktif** — ambang batas akun yang dimasukkan ke
  graf jaringan aliran dana.

## 🗂️ Struktur Tab

1. **Business Understanding** — latar belakang & tujuan proyek
2. **Data Understanding** — EDA: distribusi kelas, statistik, missing values
3. **Data Preparation** — under-sampling, encoding, train/test split
4. **Modeling** — hasil hyperparameter tuning & visualisasi pohon keputusan
5. **Evaluation** — akurasi, ROC-AUC, confusion matrix, feature importance
6. **Graph Analytics** — jaringan aliran dana, sentralitas, deteksi sindikat
7. **Kesimpulan** — ringkasan hasil & rekomendasi (dihitung otomatis dari
   data yang sedang dijalankan)

## 📝 Catatan

Semua hasil (grafik, tabel, model) dihitung ulang setiap kali tombol
**"🚀 Jalankan Analisis Lengkap"** ditekan, sesuai parameter yang dipilih
di sidebar.

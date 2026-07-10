# Dashboard Deteksi Fraud Keuangan & Analisis Jaringan Money Laundering

Dashboard Streamlit ini menampilkan **hasil akhir analisis** dari notebook
Colab Kelompok 2 (Data Mining & Analisis Jaringan Graf). Dashboard ini
bersifat **showcase / tampilan hasil** — tidak melakukan training atau
prediksi model secara langsung. Semua angka, tabel, dan grafik merepresentasikan
hasil yang sudah didapat pada notebook.

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

Tidak perlu upload dataset apa pun — dashboard langsung menampilkan hasil.

## 🗂️ Struktur Tab

1. **Business Understanding** — latar belakang & tujuan proyek
2. **Data Understanding** — distribusi kelas fraud, statistik, pola saldo tujuan
3. **Data Preparation** — ringkasan under-sampling, encoding, train/test split
4. **Modeling** — hasil hyperparameter tuning & feature importance
5. **Evaluation** — perbandingan model awal vs tuned, confusion matrix, ROC curve
6. **Graph Analytics** — ringkasan sindikat terdeteksi & ilustrasi pola jaringan
7. **Kesimpulan** — ringkasan hasil & rekomendasi

## 📝 Catatan

- Angka-angka (akurasi 99.15%, ROC-AUC 0.9933, 44 sindikat, dst.) diambil
  langsung dari hasil akhir notebook `Kelompok2_DataMining_FraudDetection.ipynb`.
- Grafik confusion matrix, ROC curve, dan visualisasi jaringan digambar ulang
  secara ilustratif mengikuti angka/pola yang dilaporkan pada notebook
  (karena dashboard tidak menyimpan dataset asli maupun model yang sudah dilatih).
- Kalau nanti kamu mau menghubungkan dashboard ini ke model/data asli (bukan
  hanya angka statis), tinggal bilang saja — bisa dikembangkan lagi.

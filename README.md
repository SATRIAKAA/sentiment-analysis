# Analisis Sentimen Ulasan

Aplikasi analisis sentimen berbasis **Streamlit** yang membandingkan tiga
pendekatan klasifikasi teks: **Linear SVM**, **Multinomial Naive Bayes**, dan
**Linear Regression ordinal**. Aplikasi dapat memproses dataset sentimen dengan
dua kelas (negatif dan positif) atau tiga kelas (negatif, netral, dan positif).

## Fitur

- Mengunggah dataset CSV dengan nama kolom yang bebas.
- Memilih kolom teks dan kolom sentimen dari sidebar.
- Menerima label teks bahasa Indonesia/Inggris atau label angka berurutan.
- Membersihkan baris duplikat dan memvalidasi isi kolom yang dipilih.
- Melatih dan membandingkan tiga model dengan pembagian data latih/uji
  terstratifikasi.
- Menampilkan distribusi kelas, akurasi, precision, recall, F1-score, dan
  confusion matrix.
- Memprediksi sentimen dari teks baru menggunakan model yang dipilih.

## Struktur proyek

```text
.
├── main.py                       # Antarmuka Streamlit
├── requirements.txt              # Dependensi Python
├── data/
│   ├── raw/dataset.csv            # Dataset contoh
│   └── processed/                 # Lokasi data hasil olahan (opsional)
├── models/                        # Lokasi model tersimpan (opsional)
├── notebooks/                     # Notebook eksplorasi (opsional)
└── src/
    └── sentiment_analysis.py      # Validasi, pipeline, pelatihan, evaluasi
```

Model dilatih saat aplikasi berjalan dan belum disimpan sebagai berkas model.
Folder `models/` tersedia untuk pengembangan penyimpanan model berikutnya.

## Persyaratan dataset

Dataset harus berupa CSV dan memiliki setidaknya dua kolom:

1. **Kolom teks**: satu ulasan/teks untuk setiap baris.
2. **Kolom sentimen**: label kelas untuk teks pada baris tersebut.

Nama kolom bebas. Pilih keduanya pada bagian **Pilih atribut dataset** di
sidebar. Dataset contoh lokal memiliki kolom `reviews` dan `label`. Dataset
tidak disertakan di repository GitHub; saat menjalankan clone baru, unggah CSV
Anda melalui sidebar. Jika berkas lokal `data/raw/dataset.csv` tersedia dan
tidak ada CSV yang diunggah, aplikasi menggunakan berkas tersebut.

### Label yang didukung

| Jumlah kelas | Label teks yang didukung | Contoh label angka |
|---|---|---|
| 2 | `negatif`/`negative`/`neg` dan `positif`/`positive`/`pos` | Dua nilai angka berbeda, misalnya `0, 1` atau `1, 2` |
| 3 | Label negatif, netral, dan positif dalam Bahasa Indonesia atau Inggris | Tiga nilai angka berbeda, misalnya `0, 1, 2` atau `1, 2, 3` |

Label angka dipetakan menurut urutan nilainya:

- Dua kelas: angka terkecil menjadi **Negatif**, angka terbesar menjadi
  **Positif**.
- Tiga kelas: angka terkecil menjadi **Negatif**, angka tengah menjadi
  **Netral**, angka terbesar menjadi **Positif**.

Pastikan pemetaan tersebut sesuai dengan arti label dalam dataset Anda. Contoh,
label biner `1 = positif` dan `0 = negatif` sudah sesuai dengan aturan di atas.
Label numerik dengan kelas yang tidak konsisten atau label teks selain alias
yang didukung akan ditolak, bukan ditebak.

Setiap kelas perlu memiliki cukup data agar dapat muncul di data latih dan data
uji. Teks dan label yang dipilih tidak boleh kosong. Baris duplikat pada kedua
kolom terpilih dihapus sebelum pelatihan.

### Contoh CSV dua kelas

```csv
ulasan,kelas
"Bahannya bagus dan nyaman",positif
"Jahitannya buruk dan mudah lepas",negatif
```

### Contoh CSV tiga kelas

```csv
komentar,sentimen
"Pengiriman cepat, produk bagus",positif
"Produk biasa saja",netral
"Barang rusak dan mengecewakan",negatif
```

Kolom tambahan, misalnya ID transaksi atau nama produk, boleh ada; aplikasi
hanya menggunakan kolom teks dan sentimen yang dipilih.

## Instalasi dan menjalankan aplikasi

Python 3.10 atau yang lebih baru direkomendasikan.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

### Windows tanpa aktivasi virtual environment

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run main.py
```

Setelah Streamlit berjalan, buka alamat lokal yang dicetak di terminal,
umumnya `http://localhost:8501`.

## Alur kerja aplikasi

1. **Membaca data**  
   Aplikasi membaca dataset contoh atau CSV yang diunggah.
2. **Memilih atribut**  
   Pilih satu kolom untuk teks dan kolom berbeda untuk label. Nama yang umum,
   misalnya `reviews`, `text`, `ulasan`, `label`, `sentiment`, dan `target`,
   akan dipilih secara otomatis bila tersedia.
3. **Validasi dan normalisasi**  
   Nilai kosong ditolak, label dinormalisasi menjadi `Negatif`, `Netral`, atau
   `Positif`, lalu duplikat pada kolom terpilih dihapus.
4. **Pembagian data**  
   Data dibagi menjadi data latih dan data uji sesuai persentase di sidebar
   (default 20%). Pembagian menggunakan stratifikasi agar proporsi kelas
   dipertahankan dan `random_state=42` agar hasil pembagian dapat diulang.
5. **Ekstraksi fitur**  
   `TfidfVectorizer` mempelajari representasi teks hanya dari data latih untuk
   evaluasi, kemudian dipasang ulang pada seluruh data untuk model prediksi.
6. **Evaluasi**  
   Ketiga model dinilai menggunakan set uji yang sama.
7. **Prediksi interaktif**  
   Aplikasi melatih model final dengan seluruh data yang telah dibersihkan,
   kemudian menggunakan model tersebut untuk prediksi teks baru.

Pipeline menggunakan TF-IDF dengan unigram dan bigram (`ngram_range=(1, 2)`),
batas maksimum 20.000 fitur, dan `sublinear_tf=True`. Untuk menghindari
kebocoran data, vectorizer berada di dalam pipeline scikit-learn dan dipelajari
hanya dari bagian latih ketika menghitung metrik.

Pengaturan TF-IDF lainnya mengikuti default scikit-learn: teks diubah menjadi
huruf kecil, tokenisasi memakai pola bawaan, dan vektor dinormalisasi dengan
norma L2. Tidak ada tahap khusus untuk menghapus stopword, menerjemahkan,
menangani negasi, memperbaiki ejaan, atau melakukan stemming/lemmatisasi.
Karena itu, kualitas dan variasi bahasa dalam dataset sangat berpengaruh.

## Model yang digunakan

Ketiga model menerima fitur TF-IDF yang sama. Teks tidak otomatis diterjemahkan
atau diberi label; model mempelajari pola dari contoh berlabel pada dataset.

### 1. Linear Support Vector Machine (SVM)

Implementasi menggunakan `LinearSVC`, sebuah SVM linear yang cocok untuk data
teks berdimensi tinggi dan sparse seperti TF-IDF. SVM mencari batas keputusan
yang memisahkan kelas dengan margin sebesar mungkin. Dengan pendekatan
one-vs-rest dari scikit-learn, model juga dapat menangani tiga kelas.

Secara sederhana, model memilih bobot `w` dan bias `b` untuk fungsi keputusan
`f(x) = w · x + b`. Parameter regularisasi `C` memakai nilai default
scikit-learn (`C=1.0`); nilai ini menyeimbangkan margin yang lebar dan penalti
untuk kesalahan pada data latih.

**Kelebihan:** cepat dan sering menjadi baseline kuat untuk klasifikasi teks.

**Catatan:** model mengeluarkan kelas, bukan probabilitas terkalibrasi.

### 2. Multinomial Naive Bayes

`MultinomialNB` menghitung kelas paling mungkin berdasarkan fitur kata dan
asumsi bahwa fitur bersifat independen satu sama lain jika kelas diketahui.
TF-IDF bernilai non-negatif sehingga cocok dengan implementasi ini. Model
menghitung probabilitas fitur per kelas dengan smoothing bawaan scikit-learn.

Secara konseptual, prediksi memilih kelas dengan probabilitas posterior terbesar:
`P(kelas | fitur) ∝ P(kelas) × ∏ P(fitur_i | kelas)`. Implementasi menggunakan
smoothing aditif dengan `alpha=1.0` (default `MultinomialNB`) agar fitur yang
belum terlihat pada suatu kelas tidak menghasilkan probabilitas nol.

**Kelebihan:** sederhana, cepat, dan efektif sebagai pembanding untuk teks.

**Catatan:** asumsi independensi kata tidak selalu sesuai dengan bahasa alami.

### 3. Linear Regression ordinal

Regresi linear bukan classifier secara bawaan. Dalam proyek ini kelas sentimen
diurutkan lalu dikodekan menjadi angka: dua kelas menjadi `0, 1`; tiga kelas
menjadi `0, 1, 2`. Regresi memprediksi skor kontinu. Skor dibulatkan ke indeks
kelas terdekat, dibatasi ke rentang kelas yang ada, lalu dikembalikan menjadi
label sentimen.

Secara sederhana, regresi memodelkan `skor = w · x + b`. Untuk dua kelas
indeksnya `0, 1`; untuk tiga kelas `0, 1, 2`. Skor kontinu dibulatkan dengan
pembulatan NumPy ke indeks terdekat, lalu indeks dibatasi ke kelas minimum dan
maksimum. Tidak ada parameter klasifikasi atau probabilitas pada model ini.

**Kelebihan:** menyediakan pembanding sederhana yang memanfaatkan urutan
negatif → netral → positif.

**Keterbatasan penting:** regresi linear mengasumsikan jarak antar-kode kelas
bermakna dan prediksi skor tidak dibatasi secara alami. Karena itu model ini
adalah eksperimen pembanding, bukan pilihan default yang selalu lebih baik
daripada classifier khusus.

### Ringkasan parameter model

| Bagian | Pengaturan |
|---|---|
| Ekstraksi teks | TF-IDF, unigram + bigram, maksimum 20.000 fitur |
| SVM | `LinearSVC` dengan parameter default, termasuk `C=1.0` |
| Naive Bayes | `MultinomialNB` dengan parameter default, termasuk `alpha=1.0` |
| Regresi | `LinearRegression` untuk nilai kelas ordinal; pembulatan dan clipping saat prediksi |
| Pembagian data | `test_size` dapat dipilih 10–40%; default 20%; `random_state=42`; stratified |
| Penalaan hyperparameter | Tidak dilakukan; tidak ada grid search atau cross-validation |

## Cara membaca metrik dan confusion matrix

- **Akurasi**: proporsi prediksi yang benar dari seluruh data uji.
- **Precision**: proporsi prediksi suatu sentimen yang benar.
- **Recall**: proporsi data suatu sentimen yang berhasil ditemukan.
- **F1-score**: rata-rata harmonik precision dan recall.
- **Confusion matrix**: baris menunjukkan kelas aktual dan kolom menunjukkan
  kelas prediksi. Nilai diagonal adalah prediksi benar; nilai di luar diagonal
  adalah kesalahan klasifikasi.

Untuk dua kelas, precision, recall, dan F1 dihitung untuk kelas **Positif**.
Untuk tiga kelas, ketiganya memakai **macro average**: metrik dihitung untuk
setiap kelas lalu dirata-ratakan tanpa membobot berdasarkan jumlah sampel.

Hasil yang ditampilkan berasal dari satu kali pembagian data latih/uji, bukan
cross-validation. Nilai metrik dapat berubah jika dataset atau proporsi uji
berubah. Skor evaluasi adalah estimasi performa dan bukan jaminan hasil pada
ulasan baru.

## Dependensi utama

- `pandas` untuk membaca dan memproses dataset.
- `numpy` untuk operasi numerik dan prediksi model regresi.
- `scikit-learn` untuk TF-IDF, model, pembagian data, dan metrik.
- `matplotlib` dan `seaborn` untuk grafik.
- `streamlit` untuk antarmuka web lokal.

Daftar dependensi lengkap ada di [`requirements.txt`](requirements.txt).

## Troubleshooting

### Kolom dataset tidak dikenali

Pastikan CSV memiliki header. Pilih kolom teks dan label yang benar dari sidebar;
nama kolom tidak harus `reviews` atau `label`.

### Aplikasi meminta dataset saat pertama dijalankan

Dataset ulasan tidak disertakan di repository. Unggah file CSV Anda melalui
sidebar sebelum menjalankan analisis.

### Label ditolak

Pastikan target berisi tepat dua atau tiga kelas yang didukung. Untuk label teks,
gunakan alias negatif/netral/positif yang tercantum di atas. Untuk label angka,
gunakan nilai numerik berurutan dan pastikan urutannya sesuai dengan kelas.

### Jumlah data terlalu sedikit

Pembagian terstratifikasi membutuhkan data yang cukup pada setiap kelas untuk
masuk ke data latih dan uji. Tambahkan sampel pada kelas yang kurang atau
tingkatkan proporsi data uji. Hindari mengandalkan skor evaluasi jika jumlah
sampel sangat kecil.

### Memulai ulang Streamlit

Hentikan server dengan `Ctrl+C`, lalu jalankan lagi:

```powershell
python -m streamlit run main.py
```

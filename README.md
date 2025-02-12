# Laporan Proyek Machine Learning - Raffi Dzaky Mahendra

## Domain Proyek
Diabetes adalah kondisi kronis yang mempengaruhi cara tubuh memproses glukosa (gula darah). Penyakit ini telah menjadi masalah kesehatan global yang serius dengan prevalensi yang terus meningkat. Menurut World Health Organization (WHO), Pada tahun 2012 saja, diabetes menyebabkan 1,5 juta kematian. Komplikasinya dapat menyebabkan serangan jantung, stroke, kebutaan, gagal ginjal, dan amputasi anggota tubuh bagian bawah.

Deteksi dini diabetes sangat penting untuk mencegah komplikasi serius seperti penyakit jantung, kebutaan, dan gagal ginjal. Machine learning dapat membantu dalam mengidentifikasi risiko diabetes pada pasien berdasarkan berbagai parameter kesehatan, memungkinkan intervensi medis yang lebih awal dan efektif.

**Referensi**:
- [Global Report on Diabetes - World Health Organization](https://www.who.int/publications/i/item/9789241565257)


## Business Understanding

### Problem Statements
1. Bagaimana cara mengembangkan model machine learning yang dapat memprediksi risiko diabetes pada pasien berdasarkan parameter kesehatan mereka?
2. Apa saja faktor-faktor kesehatan yang paling berpengaruh dalam prediksi diabetes?
3. Seberapa akurat model dapat membedakan antara pasien yang berisiko diabetes dan yang tidak?

### Goals
1. Mengembangkan model klasifikasi yang dapat memprediksi risiko diabetes dengan tingkat akurasi yang tinggi.
2. Mengidentifikasi parameter kesehatan yang memiliki pengaruh signifikan terhadap diagnosis diabetes.
3. Mencapai metrik evaluasi (accuracy, precision, recall, dan F1-score) yang seimbang untuk menghindari kesalahan diagnosis.

### Solution statements
1. Menggunakan algoritma Random Forest Classifier dengan teknik hyperparameter tuning melalui Grid Search CV untuk mengoptimalkan performa model.
2. Menerapkan preprocessing data yang komprehensif, termasuk:
    - Penanganan missing values dengan pendekatan median berdasarkan kelompok outcome
    - Penghapusan outlier menggunakan metode IQR
    - Standardisasi fitur menggunakan StandardScaler

## Data Understanding
Dataset yang digunakan adalah Pima Indians Diabetes Dataset yang berisi informasi medis dari pasien wanita dengan keturunan Indian Pima. Dataset ini terdiri dari 768 sampel dengan 8 fitur dan 1 target variable. [Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset/data)

### Variabel-variabel pada diabetes dataset adalah sebagai berikut:
- Pregnancies: Untuk menyatakan jumlah kehamilan
- Glucose: Untuk mengekspresikan tingkat Glukosa dalam darah
- BloodPresure: Untuk mengekspresikan pengukuran tekanan darah
- SkinThickness: Untuk mengekspresikan ketebalan kulit
- Insulin: Untuk mengekspresikan tingkat Insulin dalam darah
- BMI: Untuk mengekspresikan indeks masa tubuh

- DiabetesPedigreeFunction: untuk mengungkapkan persentase Diabetes

- Age: Untuk mengekspresikan usia

- Outcome: Untuk menyatakan hasil akhir 1 adalah Ya dan 0 adalah Tidak

**Exploratory Data Analysis (EDA)**:
1. Analisis Distribusi Diabetes Outcome:
    - Dari 582 record, terdapat 396 orang yang negatif dan 186 orang positif diabetes

2. Distribusi Glucose berdasarkan Outcome:
    - Pasien dengan diabetes (Outcome=1) cenderung memiliki kadar glukosa yang lebih tinggi
    - Nilai glukosa di atas 140 lebih sering ditemukan pada pasien diabetes

3. Distribusi BMI berdasarkan Outcome:
    - Pasien diabetes memiliki rata-rata BMI yang sedikit lebih tinggi
    - Kedua kelompok menunjukkan distribusi yang cenderung normal

4. Distribusi Age berdasarkan Outcome:
    - Dataset didominasi oleh pasien berusia muda (20-30 tahun)
    - Proporsi pasien diabetes relatif meningkat pada kelompok usia yang lebih tua

5. Distribusi BloodPresure berdasarkan Outcome:
    - Tekanan darah antara 60-80 paling umum ditemui
    - Tidak ada perbedaan yang sangat signifikan dalam distribusi tekanan darah antara pasien diabetes dan non-diabetes

## Data Preparation
Beberapa teknik data preparation yang diterapkan:
1. Penanganan nilai nol yang tidak valid:
    - Mengidentifikasi fitur dengan nilai nol yang tidak masuk akal
    - Mengganti nilai nol dengan median berdasarkan fitur 'outcome'
    - Pendekatan ini akan mempertahankan distribusi data
2. Penanganan outlier:
    - Menggunakan metode IQR (Interquartile Range)
    - Menghapus data di luar range (Q1 - 1.5IQR) hingga (Q3 + 1.5IQR)
    - Proses ini mengurangi jumlah data dari 768 menjadi 582 sampel (24.22% data dihapus)
3. Pembagian Dataset:
    - Memisahkan antara fitur dan label
    - Membagi data menjadi training set (80%) dan testing set (20%)
    - Jumlah data training: 465 samples
    - Jumlah data testing: 117 samples
4. Standardisasi fitur:
    - Menggunakan StandardScaler untuk menormalkan skala fitur
    
## Modeling
Dalam proyek ini, menggunakan algoritma Random Forest Classifier dengan teknik Grid Search CV untuk optimasi hyperparameter. Random Forest dipilih karena:
- Mampu menangani data numerik dengan baik
- Dapat menangani hubungan non-linear antara fitur
- Mengurangi risiko overfitting

Hyperparameter yang dioptimasi:
- n_estimators: [50, 100, 200]
- max_depth: [None, 10, 20, 30]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

Hasil Grid Search menghasilkan hyperparameter terbaik:
- n_estimators: 200
- max_depth: None
- min_samples_split: 10
- min_samples_leaf: 4

## Evaluation
Model dievaluasi menggunakan beberapa metrik:
1. Accuracy (0.8889)
    - Mengukur proporsi prediksi yang benar dari total prediksi
    - Menunjukkan bahwa model memprediksi dengan benar 88.89% kasus
2. Precision (0.8788)
    - Mengukur proporsi prediksi positif yang benar
    - 87.88% pasien yang diprediksi memiliki diabetes benar-benar memiliki diabetes
3. Recall (0.7632):
    - Mengukur proporsi kasus positif yang berhasil diprediksi
    - Model berhasil mengidentifikasi 76.32% dari total pasien diabetes yang sebenarnya
4. F1-Score (0.8169):
    - Menyeimbangkan trade-off antara precision dan recall
    - Score yang cukup baik mengindikasikan model yang seimbang

Fomula metrik evaluasi
```Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
```

- TP (True Positive): Pasien diabetes yang benar diprediksi memiliki diabetes
- TN (True Negative): Pasien non-diabetes yang benar diprediksi tidak memiliki diabetes
- FP (False Positive): Kasus non-diabetes yang diprediksi sebagai diabetes
- FN (False Negative): Kasus diabetes yang diprediksi sebagai non-diabetes

Confusson Matrix
<div align="center">
  <img src="image/Confusion Matrix.png" style="max-width: 50%; height: auto; margin: 10px;">
</div>




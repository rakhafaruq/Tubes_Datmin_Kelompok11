
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr_model = joblib.load("model_logres.joblib")
kmeans_model = joblib.load("model_kmeans.joblib")
df = pd.read_csv("heart_cleveland_upload.csv")

X_kmeans = df[['age', 'thalach']]
df['cluster'] = kmeans_model.predict(X_kmeans)

x = df.drop(columns=['condition', 'cluster'])
y = df['condition']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

lr_model.fit(X_train_resampled, y_train_resampled)

y_pred = lr_model.predict(X_test)

y_proba_test = lr_model.predict_proba(X_test)[:, 1]
y_pred_threshold = (y_proba_test >= 0.6).astype(int)

st.title("🫀 Dashboard Analisis Penyakit Jantung")

tab1, tab2, tab3 = st.tabs(["📊 Dataset", "📍 Clustering", "🧪 Prediksi"])

with tab1:
    st.subheader("Informasi Dataset")
    st.write(df.head())
    st.write("Jumlah Data:", df.shape[0])
    st.write("Jumlah Fitur:", df.shape[1])
    condition_labels = df['condition'].replace({0: '✅ Sehat', 1: '⚠️ Berisiko'})
    st.bar_chart(condition_labels.value_counts())


with tab2:
    st.subheader("Visualisasi Clustering Berdasarkan Usia dan Denyut Jantung")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='age', y='thalach', hue='cluster', palette='viridis', ax=ax)
    plt.title("Clustering Berdasarkan Umur dan Thalach")
    st.pyplot(fig)

    st.write("Pembagian Pasien Berdasarkan Thalach per Cluster")

    cluster_0_min = df[df['cluster'] == 0]['thalach'].min()
    cluster_0_max = df[df['cluster'] == 0]['thalach'].max()

    cluster_1_min = df[df['cluster'] == 1]['thalach'].min()
    cluster_1_max = df[df['cluster'] == 1]['thalach'].max()

    st.write(f"Cluster 0: Rentang thalach {cluster_0_min} hingga {cluster_0_max}")
    st.write(f"Cluster 1: Rentang thalach {cluster_1_min} hingga {cluster_1_max}")

    cluster_0 = len(df[df['cluster'] == 0])
    cluster_1 = len(df[df['cluster'] == 1])

    cluster_summary = pd.DataFrame({
        "Denyut Jantung": ["147 - 202", "71 - 147"],
        "Jumlah Pasien": [cluster_0, cluster_1]
    })

    st.dataframe(cluster_summary)

with tab3:
    st.subheader("Prediksi Penyakit Jantung (Logistic Regression)")

    age = st.slider("Usia", 29, 77, 50)
    sex = st.selectbox("Jenis Kelamin", options=["Perempuan", "Laki-laki"])
    sex = 0 if sex == "Perempuan" else 1

    cp_dict = {
        "Typical Angina (biasanya bagian dada kiri atau tengah)": 0,
        "Atypical Angina (nyeri dada yang tidak biasa)": 1,
        "Non-anginal Pain (bukan disebabkan masalah jantung)": 2,
        "Asymptomatic (tanpa gejala)": 3
    }
    cp_label = st.selectbox("Tipe Nyeri Dada", list(cp_dict.keys()))
    cp = cp_dict[cp_label]

    trestbps = st.slider("Tekanan Darah Istirahat (mm Hg)", 94, 200, 120)
    chol = st.slider("Kolesterol (mg/dl)", 126, 564, 240)

    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl?", options=["Tidak", "Ya"])
    fbs = 1 if fbs == "Ya" else 0

    restecg_dict = {
        "Normal": 0,
        "Kelainan gelombang ST-T": 1,
        "Hipertrofi Ventrikel Kiri": 2
    }
    restecg_label = st.selectbox("Hasil ECG Saat Istirahat", list(restecg_dict.keys()))
    restecg = restecg_dict[restecg_label]

    thalach = st.slider("Detak Jantung Maksimal (thalach)", 71, 202, 150)

    exang = st.selectbox("Mengalami Angina Saat Latihan?", options=["Tidak", "Ya"])
    exang = 1 if exang == "Ya" else 0

    oldpeak = st.slider("Depresi ST (oldpeak)", 0.0, 6.2, 1.0)

    slope_dict = {
        "Meningkat": 0,
        "Datar": 1,
        "Menurun": 2
    }
    slope_label = st.selectbox("Kemiringan ST", list(slope_dict.keys()))
    slope = slope_dict[slope_label]

    ca = st.selectbox("Jumlah Pembuluh Darah Besar yang Terlihat (0-3)", options=["0", "1", "2", "3"])
    ca = int(ca)

    thal_dict = {
        "Normal": 0,
        "Cacat Tetap": 1,
        "Cacat Reversibel": 2
    }
    thal_label = st.selectbox("Hasil Tes Thalium", list(thal_dict.keys()))
    thal = thal_dict[thal_label]


    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]],
                                columns=['age', 'sex', 'cp', 'trestbps', 'chol',
                                        'fbs', 'restecg', 'thalach', 'exang',
                                        'oldpeak', 'slope', 'ca', 'thal'])

    if st.button("Prediksi"):
        proba = lr_model.predict_proba(input_data)[0][1]
        prediction = (proba >= 0.6).astype(int)
        st.write(f"**Hasil Prediksi:** {'⚠️ Berisiko' if prediction else '✅ Sehat'}")
        st.write(f"**Probabilitas:** {proba:.2f}")

    st.subheader("Confusion Matrix")

    y_pred = lr_model.predict(X_test)
    y_proba_test = lr_model.predict_proba(X_test)[:, 1]
    y_pred_threshold = (y_proba_test >= 0.6).astype(int)

    cm = confusion_matrix(y_test, y_pred_threshold)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Sehat', 'Berisiko'], yticklabels=['Sehat', 'Berisiko'])
    ax_cm.set_xlabel('Prediksi')
    ax_cm.set_ylabel('Aktual')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

    st.subheader("ROC Curve (Training Data)")

    y_proba = lr_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig_roc)


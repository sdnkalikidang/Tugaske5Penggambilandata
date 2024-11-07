# Install dependencies jika diperlukan
# pip install pandas
# pip install numpy
# pip install scikit-learn
# pip install streamlit

import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('pencemaran_udara.csv')

# Memisahkan fitur (X) dan target (y)
X = df[['pm10', 'so2', 'co', 'o3', 'no2']].fillna(0)  # Pilih kolom fitur dan isi nilai NaN dengan 0
y = df['categori']  # Kolom target

# Split data ke dalam data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Training model dengan RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Evaluasi akurasi model
y_pred = classifier.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {score}')

# Menyimpan model ke file
with open("air_quality_classifier.pkl", "wb") as pickle_out:
    pickle.dump(classifier, pickle_out)

# Load model untuk prediksi
pickle_in = open('air_quality_classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


# Fungsi prediksi menggunakan model
def prediction(pm10, so2, co, o3, no2):
    # Predict menggunakan data input
    pred = classifier.predict([[pm10, so2, co, o3, no2]])
    return pred

# Streamlit
def main():
    # Judul halaman
    st.title("Air Quality Prediction")

    # Desain halaman dengan HTML
    html_temp = """
    <div style="background-color:lightblue;padding:13px">
    <h1 style="color:black;text-align:center;">Streamlit Air Quality Classifier ML App</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input dari user
    pm10 = st.number_input("PM10", value=0)
    so2 = st.number_input("SO2", value=0)
    co = st.number_input("CO", value=0)
    o3 = st.number_input("O3", value=0)
    no2 = st.number_input("NO2", value=0)

    result = ""

    # Tombol prediksi
    if st.button("Predict"):
        prediction_result = prediction(pm10, so2, co, o3, no2)
        result = prediction_result[0]  # Mengambil hasil prediksi

    st.success(f'The air quality is classified as: {result}')

if __name__ == '__main__':
    main()

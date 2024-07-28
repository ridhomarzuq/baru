import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Memuat data
data = pd.read_csv('heart_2020_cleaned1_final.csv')

# Memisahkan fitur dan target
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fungsi untuk login
def login(username, password):
    if username == 'user' and password == 'pass':
        return True
    else:
        return False

# Halaman Login
def login_page():
    st.title('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        if login(username, password):
            st.session_state['logged_in'] = True
            st.success('Login berhasil')
        else:
            st.error('Username atau password salah')

# Halaman Prediksi
def prediction_page():
    st.title('Prediksi Penyakit Jantung')
    st.write('Masukkan data untuk prediksi:')
    
    # Membuat placeholder input data
    Smoking = st.number_input('Smoking (0 atau 1)', min_value=0, max_value=1, value=0)
    AlcoholDrinking = st.number_input('Alcohol Drinking (0 atau 1)', min_value=0, max_value=1, value=0)
    Stroke = st.number_input('Stroke (0 atau 1)', min_value=0, max_value=1, value=0)
    Sex = st.number_input('Sex (0 untuk Female, 1 untuk Male)', min_value=0, max_value=1, value=0)
    AgeCategory = st.number_input('Age Category (15.0 - 80.0)', min_value=15.0, max_value=80.0, value=50.0)
    Diabetic = st.number_input('Diabetic (0 atau 1 atau 2)', min_value=0, max_value=2, value=0)
    SleepTime = st.number_input('Sleep Time (0.0 - 24.0)', min_value=0.0, max_value=24.0, value=7.0)
    
    # Prediksi
    if st.button('Prediksi'):
        input_data = pd.DataFrame({
            'Smoking': [Smoking],
            'AlcoholDrinking': [AlcoholDrinking],
            'Stroke': [Stroke],
            'Sex': [Sex],
            'AgeCategory': [AgeCategory],
            'Diabetic': [Diabetic],
            'SleepTime': [SleepTime]
        })
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.write('Hasil prediksi: Positive Heart Disease')
        else:
            st.write('Hasil prediksi: Negative Heart Disease')

# Fungsi utama
def main():
    st.set_page_config(page_title='Prediksi Penyakit Jantung', layout='wide', initial_sidebar_state='collapsed')
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if not st.session_state['logged_in']:
        login_page()
    else:
        prediction_page()

if __name__ == '__main__':
    main()

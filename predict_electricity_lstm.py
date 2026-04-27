import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="توقع تكلفة الكهرباء", layout="wide")

@st.cache_resource
def load_all():
    mlp = joblib.load('mlp_doubled_neurons_model.joblib')
    scaler = joblib.load('scaler.joblib')
    lstm = tf.keras.models.load_model('multi_output_lstm_model.h5', compile=False)
    return mlp, scaler, lstm

try:
    mlp_model, scaler, lstm_model = load_all()
    st.sidebar.success("✅ الملفات جاهزة")
except Exception as e:
    st.sidebar.error("❌ تأكدي من الملفات")

st.title("⚡ نظام إدارة طاقة المدينة الذكية")
st.markdown("---")

# تقسيم الـ 12 مدخل اللي في الإكسيل
col1, col2, col3 = st.columns(3)

with col1:
    f1 = st.number_input("Site Area:", value=2000.0)
    f2 = st.number_input("Water Consumption:", value=4000.0)
    f3 = st.number_input("Recycling Rate:", value=20.0)
    f4 = st.number_input("Utilisation Rate:", value=50.0)

with col2:
    f5 = st.number_input("Air Quality Index:", value=70.0)
    f6 = st.number_input("Issue Resolution:", value=60.0)
    f7 = st.number_input("Resident Satisfaction:", value=80.0)
    f8 = st.number_input("Carbon Emissions:", value=100.0)

with col3:
    # تحويل الاختيارات لأرقام (0 أو 1)
    f9 = st.selectbox("Structure Type 1:", [0, 1])
    f10 = st.selectbox("Structure Type 2:", [0, 1])
    f11 = st.selectbox("Structure Type 3:", [0, 1])
    f12 = st.number_input("Electricity Consumption:", value=1500.0)

if st.button("🚀 توقع التكلفة الآن"):
    try:
        # هنا الحل: بنجمع الـ 12 اللي دخلتيهم + صفر زيادة عشان نكملهم 13 للميزان
        raw_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, 0.0]])
        
        # 1. الميزان (Scaler)
        scaled_data = scaler.transform(raw_data)
        
        # 2. التوقع (LSTM) بياخد الـ 13 كاملين بشكل ثلاثي الأبعاد
        lstm_input = scaled_data.reshape(1, 1, 13)
        prediction = lstm_model.predict(lstm_input)
        
        res = prediction[0][0]
        st.success(f"### التكلفة المتوقعة: {abs(res):.2f} دولار")
        st.balloons()
        
    except Exception as e:
        st.error(f"حدث خطأ: {e}")

st.markdown("---")
st.caption("مشروع التخرج 2026 - ى")

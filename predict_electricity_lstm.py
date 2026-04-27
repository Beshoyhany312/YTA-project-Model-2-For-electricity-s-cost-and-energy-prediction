import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="نظام إدارة طاقة المدينة الذكية", layout="wide")

@st.cache_resource
def load_all():
    mlp = joblib.load('mlp_doubled_neurons_model.joblib')
    scaler = joblib.load('scaler.joblib')
    lstm = tf.keras.models.load_model('multi_output_lstm_model.h5', compile=False)
    return mlp, scaler, lstm

mlp_model, scaler, lstm_model = load_all()

st.title("⚡ نظام التوقع الذكي لتكاليف الكهرباء")
st.markdown("---")

# تقسيم المدخلات على أعمدة عشان المنظر يكون احترافي
col1, col2, col3 = st.columns(3)

with col1:
    f1 = st.number_input("Site Area (sq meters):", value=2000.0)
    f2 = st.number_input("Water Consumption:", value=4000.0)
    f3 = st.number_input("Recycling Rate (%):", value=20.0)
    f4 = st.number_input("Utilisation Rate:", value=50.0)

with col2:
    f5 = st.number_input("Air Quality Index:", value=70.0)
    f6 = st.number_input("Issue Resolution (%):", value=60.0)
    f7 = st.number_input("Resident Satisfaction:", value=80.0)
    f8 = st.number_input("Carbon Emissions:", value=100.0)

with col3:
    # المدخلات المنطقية (True/False) هنخليها Selectbox
    f9 = st.selectbox("Structure Type 1:", [0, 1])
    f10 = st.selectbox("Structure Type 2:", [0, 1])
    f11 = st.selectbox("Structure Type 3:", [0, 1])
    f12 = st.number_input("Electricity Consumption (kWh):", value=1500.0)

st.markdown("---")
predict_btn = st.button("🚀 تحليل البيانات وتوقع التكلفة")

if predict_btn:
    try:
        # تجميع المدخلات الـ 12 + خانة إضافية لأن الميزان (Scaler) متدرب على 13
        # (بناءً على الخطأ السابق، الـ Scaler محتاج 13 عمود)
        input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, 0.0]])
        
        # تحويل البيانات باستخدام الميزان
        scaled_data = scaler.transform(input_data)
        
        # اختيار الموديل (مثلاً LSTM)
        # بنجهز البيانات بشكل (1, 1, 13) كما طلب الموديل في الصورة
        lstm_input = scaled_data.reshape(1, 1, 13)
        prediction = lstm_model.predict(lstm_input)[0][0]
        
        st.success(f"### التكلفة المتوقعة شهرياً: {abs(prediction):.2f} دولار")
        st.info("تم الحساب بناءً على معايير الاستدامة وجودة الحياة المتوفرة.")
        st.balloons()
        
    except Exception as e:
        st.error(f"حدث خطأ أثناء الحساب: {e}")

st.caption("مشروع تخرج 2026")

import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# 1. إعدادات الصفحة
st.set_page_config(page_title="توقع الطاقة الذكي", page_icon="⚡", layout="wide")

# 2. وظيفة تحميل الموديلات
@st.cache_resource
def load_all_assets():
    mlp = joblib.load('mlp_doubled_neurons_model.joblib')
    scaler = joblib.load('scaler.joblib')
    lstm = tf.keras.models.load_model('multi_output_lstm_model.h5', compile=False)
    return mlp, scaler, lstm

try:
    mlp_model, scaler, lstm_model = load_all_assets()
    st.sidebar.success("✅ الأنظمة جاهزة")
except Exception as e:
    st.sidebar.error("❌ مشكلة في الملفات")

# 3. واجهة المستخدم (تصميم جديد ومضمون)
st.title("🤖 نظام توقع استهلاك الطاقة الكهربائية")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.info("### ⚙️ الإعدادات")
    model_choice = st.radio(
        "اختر تقنية الذكاء الاصطناعي:",
        ("الموديل الأساسي (MLP)", "الموديل المتطور (LSTM)")
    )
    input_val = st.number_input("كمية الاستهلاك (kWh):", min_value=0.0, value=100.0)
    predict_btn = st.button("🚀 تحليل وتوقع النتيجة")

with col2:
    st.success("### 📊 نتائج التحليل")
    if predict_btn:
        try:
            if model_choice == "الموديل الأساسي (MLP)":
                data = np.zeros((1, 10))
                data[0, 0] = input_val
                res = mlp_model.predict(scaler.transform(data))[0][0]
            else:
                data = np.zeros((1, 1, 10))
                data[0, 0, 0] = input_val
                res = lstm_model.predict(data)[0][0]

            # عرض النتيجة بشكل كبير وواضح
            st.metric(label=f"التكلفة المتوقعة ({model_choice})", value=f"{res:.2f} جنيه")
            st.balloons()
            
        except Exception as e:
            st.error(f"حدث خطأ: {e}")
    else:
        st.write("انتظار إدخال البيانات...")

st.markdown("---")
st.caption("تم التطوير بواسطة المندسة سلمى - مشروع التخرج 2026")

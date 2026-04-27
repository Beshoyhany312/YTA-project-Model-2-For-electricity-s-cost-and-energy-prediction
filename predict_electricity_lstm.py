import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# 1. إعدادات الصفحة بشكل احترافي
st.set_page_config(
    page_title="توقع الطاقة الذكي",
    page_icon="⚡",
    layout="wide"
)

# إضافة لمسة جمالية بالـ CSS (ألوان وتنسيق)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 5px solid #007bff;
    }
    </style>
    """, unsafe_allow_stdio=True)

# 2. وظيفة تحميل الموديلات
@st.cache_resource
def load_all_assets():
    mlp = joblib.load('mlp_doubled_neurons_model.joblib')
    scaler = joblib.load('scaler.joblib')
    lstm = tf.keras.models.load_model('multi_output_lstm_model.h5', compile=False)
    return mlp, scaler, lstm

# التحميل وعرض الحالة في الجنب بشكل هادئ
try:
    mlp_model, scaler, lstm_model = load_all_assets()
    st.sidebar.success("✅ الأنظمة جاهزة للعمل")
except Exception as e:
    st.sidebar.error("❌ عذراً، هناك مشكلة في تحميل البيانات")

# 3. تصميم واجهة المستخدم
st.title("🤖 نظام توقع استهلاك الطاقة الكهربائية")
st.markdown("---")

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("⚙️ إعدادات المدخلات")
    model_choice = st.radio(
        "اختر تقنية الذكاء الاصطناعي:",
        ("الموديل الأساسي (MLP)", "الموديل المتطور (LSTM)")
    )
    
    input_val = st.number_input("كمية الاستهلاك الحالية (kWh):", min_value=0.0, value=100.0, step=1.0)
    predict_btn = st.button("تحليل وتوقع النتيجة")

with col2:
    st.subheader("📊 نتائج التحليل")
    if predict_btn:
        with st.spinner('جاري الحساب...'):
            try:
                if model_choice == "الموديل الأساسي (MLP)":
                    data = np.zeros((1, 10))
                    data[0, 0] = input_val
                    scaled_data = scaler.transform(data)
                    prediction = mlp_model.predict(scaled_data)
                    final_result = prediction[0][0]
                else:
                    data = np.zeros((1, 1, 10))
                    data[0, 0, 0] = input_val
                    prediction = lstm_model.predict(data)
                    final_result = prediction[0][0]

                # عرض النتيجة بشكل مبهر
                st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: #555;">التكلفة المتوقعة باستخدام {model_choice}</h3>
                        <h1 style="color: #007bff; font-size: 50px;">{final_result:.2f}</h1>
                        <p style="color: #888;">جنيه مصري</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.balloons() # حركة احتفالية بسيطة
                
            except Exception as e:
                st.error(f"حدث خطأ فني: {e}")
    else:
        st.info("قم بإدخال البيانات والضغط على زر التحليل لمشاهدة التوقعات.")

# تذييل الصفحة
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.caption("مشروع تخرج")

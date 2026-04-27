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
    st.sidebar.success("✅ الأنظمة جاهزة للعمل")
except Exception as e:
    st.sidebar.error("❌ مشكلة في تحميل الملفات")

# 3. واجهة المستخدم
st.title("🤖 نظام توقع استهلاك الطاقة الكهربائية")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.info("### ⚙️ الإعدادات")
    model_choice = st.radio("اختر تقنية الذكاء الاصطناعي:", ("الموديل الأساسي (MLP)", "الموديل المتطور (LSTM)"))
    input_val = st.number_input("كمية الاستهلاك (kWh):", min_value=0.0, value=100.0)
    predict_btn = st.button("🚀 تحليل وتوقع النتيجة")

with col2:
    st.success("### 📊 نتائج التحليل")
    if predict_btn:
        try:
            # الحل الجذري لكل مشاكل الـ Shapes
            # 1. بنشوف الـ Scaler محتاج كام عمود (بناءً على ملف scaler.joblib)
            n_scaler = scaler.n_features_in_
            data_for_sc = np.zeros((1, n_scaler))
            data_for_sc[0, 0] = input_val
            scaled_full = scaler.transform(data_for_sc)
            
            if "MLP" in model_choice:
                # الموديل الأساسي MLP
                n_mlp = mlp_model.input_shape[1] if hasattr(mlp_model, 'input_shape') else 10
                final_input = np.zeros((1, n_mlp))
                # بناخد اللي نقدر عليه من البيانات الموزونة
                fill_len = min(scaled_full.shape[1], n_mlp)
                final_input[0, :fill_len] = scaled_full[0, :fill_len]
                res = mlp_model.predict(final_input)[0][0]
            else:
                # الموديل المتطور LSTM (اللي كان طالب 13 عمود في الصورة)
                # بنقرأ الـ Shape المطلوب من الموديل نفسه
                target_shape = lstm_model.input_shape # (None, 1, 13)
                n_lstm = target_shape[-1] 
                
                lstm_input = np.zeros((1, 1, n_lstm))
                # بنملا الأعمدة المتاحة
                fill_len = min(scaled_full.shape[1], n_lstm)
                lstm_input[0, 0, :fill_len] = scaled_full[0, :fill_len]
                
                res = lstm_model.predict(lstm_input)[0][0]

            st.metric(label=f"التكلفة المتوقعة ({model_choice})", value=f"{abs(res):.2f} جنيه")
            st.balloons()
            
        except Exception as e:
            st.error(f"خطأ تقني: {e}")
    else:
        st.write("أدخل البيانات ثم اضغط على الزر...")

st.markdown("---")
st.caption("  مشروع التخرج 2026")

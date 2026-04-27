import streamlit as st
import pandas as pd
import numpy as np
import joblib

# تحميل الموديل (تأكدي من اسم الملف)
model = joblib.load('electricity_model.pkl')

st.title("حساب تكلفة الكهرباء بناءً على كافة الخصائص")

st.subheader("أدخل بيانات الموقع:")

# إنشاء أعمدة لتنظيم الواجهة
col1, col2 = st.columns(2)

with col1:
    site_area = st.number_input("Site Area (square meters)", value=0.0)
    water_cor = st.number_input("Water Consumption", value=0.0)
    recycling = st.number_input("Recycling Rate", value=0.0)
    utilisation = st.number_input("Utilisation", value=0.0)
    air_quality = st.number_input("Air Quality Issue", value=0.0)

with col2:
    resource = st.number_input("Resource Usage", value=0.0)
    resident_c = st.number_input("Resident Count", value=0.0)
    structure_1 = st.number_input("Structure Type 1", value=0.0)
    structure_2 = st.number_input("Structure Type 2", value=0.0)
    structure_3 = st.number_input("Structure Type 3", value=0.0)

# زر التوقع
if st.button("توقع التكلفة (Cost)"):
    # تجميع المدخلات في مصفوفة بنفس ترتيب أعمدة الداتا سيت
    features = np.array([[site_area, water_cor, recycling, utilisation, 
                          air_quality, resource, resident_c, 
                          structure_1, structure_2, structure_3]])
    
    # عمل التوقع
    prediction = model.predict(features)
    
    st.success(f"التكلفة المتوقعة للكهرباء هي: {prediction[0]:,.2f}")

# خيار إضافي: رفع ملف كامل للتوقع
st.divider()
st.subheader("أو ارفع ملف CSV للتوقع الجماعي")
uploaded_file = st.file_uploader("اختر ملف البيانات", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if st.button("توقع للملف بالكامل"):
        results = model.predict(df)
        df['Predicted_Electricity_Cost'] = results
        st.write(df)

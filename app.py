import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Tải scaler và mô hình
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
model = load_model("Artificial_Neural_Network_Case_Study.h5")

# Giao diện ứng dụng
st.title("Dự đoán khách hàng có rời đi hay không")
st.write("Nhập thông tin khách hàng để dự đoán")

# Nhập dữ liệu đầu vào
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Số năm gắn bó)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_of_products = st.number_input("Số sản phẩm sử dụng", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Có thẻ tín dụng?", ["Có", "Không"])
is_active_member = st.selectbox("Là thành viên hoạt động?", ["Có", "Không"])
estimated_salary = st.number_input("Mức lương ước tính", min_value=0.0, value=100000.0)
gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
geography = st.selectbox("Khu vực", ["France", "Germany", "Spain"])

# Xử lý đầu vào
gender = 1 if gender == "Nam" else 0
has_cr_card = 1 if has_cr_card == "Có" else 0
is_active_member = 1 if is_active_member == "Có" else 0
geography_france, geography_germany, geography_spain = 0, 0, 0
if geography == "Germany":
    geography_germany = 1
elif geography == "Spain":
    geography_spain = 1

# Dự đoán nếu nhấn nút
if st.button("Dự đoán"):
    input_data = np.array([[credit_score, age, tenure, balance, num_of_products, has_cr_card,is_active_member, estimated_salary, geography_germany, geography_spain]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    result = "Khách hàng sẽ rời đi" if prediction > 0.5 else "Khách hàng sẽ ở lại"
    st.subheader(f"Kết quả: {result}")

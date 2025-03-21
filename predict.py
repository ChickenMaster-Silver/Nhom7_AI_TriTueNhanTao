import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# 1. Đọc dữ liệu đầu vào
df = pd.read_csv("Artificial_Neural_Network_Case_Study_data.csv")

# 2. Xử lý dữ liệu
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

X = df.drop(columns=["Exited"])

# 3. Tải scaler và chuẩn hóa dữ liệu
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
X_scaled = scaler.transform(X)

# 4. Tải mô hình
model = load_model("Artificial_Neural_Network_Case_Study.h5")

# 5. Dự đoán
y_pred = (model.predict(X_scaled) > 0.5).astype(int)

# 6. Lưu kết quả
df_result = pd.DataFrame({"Actual": df["Exited"], "Predicted": y_pred.flatten()})
df_result.to_csv("DuDoan.csv", index=False)
print("Dự đoán đã được lưu vào DuDoan.csv")

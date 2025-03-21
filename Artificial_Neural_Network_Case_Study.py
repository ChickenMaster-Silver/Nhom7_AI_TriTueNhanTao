import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Đọc dữ liệu
df = pd.read_csv("D:\BaiTap\TriTueNhanTao\BaiTap\BTL\Artificial_Neural_Network_Case_Study_data.csv")

# 2. Xử lý dữ liệu
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

X = df.drop(columns=["Exited"])
y = df["Exited"]

# 3. Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lưu scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 5. Xây dựng mô hình
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Huấn luyện
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 7. Lưu mô hình
model.save("bank_churn_model.h5")
print("Mô hình đã được lưu thành công!")
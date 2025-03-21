Đề tài của bọn em là về dụe đoán khách hàng rời bỏ ngân hàng bằng mô hình mạng NEURON

kiến trúc của mô hình mạng neuron (model.summary()) trong Keras

![image](https://github.com/user-attachments/assets/8baab799-f3d5-425b-9a27-f573996ed706)

 Tính số tham số của từng lớp:
Lớp Dense đầu tiên (dense):
  Đầu vào có 11 đặc trưng → (11 * 32) + 32 = 384 tham số (trọng số + bias).
Lớp Dense 2 (dense_1):
  Đầu vào có 32 nơ-ron → (32 * 16) + 16 = 528 tham số.
Lớp Dense 3 (dense_2):
  Đầu vào có 16 nơ-ron → (16 * 8) + 8 = 136 tham số.
Lớp Output (dense_3):
  Đầu vào có 8 nơ-ron → (8 * 1) + 1 = 9 tham số.
Điểm đáng chú ý:
- Dropout không có tham số vì nó chỉ tắt ngẫu nhiên nơ-ron trong quá trình huấn luyện.
- Số nơ-ron giảm dần (32 → 16 → 8 → 1) giúp mô hình tổng quát tốt hơn.
- Sử dụng ReLU trong các lớp ẩn và Sigmoid trong lớp đầu ra (phù hợp với bài toán phân loại nhị phân).

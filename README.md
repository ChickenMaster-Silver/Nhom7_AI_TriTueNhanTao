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

Kết quả của quá trình đào tạo của mô hình mạng neuron nhân tạo (ANN).

![image](https://github.com/user-attachments/assets/5993fa33-4acc-4701-b6c7-4f21bbdd6202)
![image](https://github.com/user-attachments/assets/ac7fc257-b56b-40c6-b9ee-1fb23e631ec3)


Phân tích kết quả huấn luyện
Accuracy (độ chính xác trên tập huấn luyện):

Bắt đầu từ 0.7691 ở epoch 1.
Tăng dần lên 0.8536 ở epoch 22.
Validation Accuracy (độ chính xác trên tập kiểm tra):

Bắt đầu từ 0.8035 ở epoch 1.
Dần tăng lên 0.8610 ở epoch 22.
Loss (hàm mất mát trên tập huấn luyện):

Giảm dần từ 0.5577 (epoch 1) xuống 0.3503 (epoch 22).
Validation Loss (hàm mất mát trên tập kiểm tra):

Giảm từ 0.4547 xuống 0.3407, cho thấy mô hình đang học tốt.

Báo cáo đánh giá mô hình

![image](https://github.com/user-attachments/assets/1de4b17d-e85f-45e0-b8ef-b820ff5d2bfe)

- Lớp 0 (không rời dịch vụ)
Precision: 0.87 → Trong tất cả các dự đoán là 0, có 87% là đúng.
Recall: 0.96 → Trong tất cả các mẫu thực tế là 0, mô hình dự đoán đúng 96%.
F1-score: 0.91 → Hiệu suất tổng thể tốt.
- Lớp 1 (rời dịch vụ - khách hàng cần dự đoán)
Precision: 0.72 → Trong tất cả các dự đoán là 1, có 72% là đúng.
Recall: 0.42 → Trong tất cả các khách hàng thực sự rời dịch vụ, mô hình chỉ tìm đúng 42%.
F1-score: 0.53 → Hiệu suất chưa tốt, cần cải thiện.
- Tổng quan mô hình
Accuracy (Độ chính xác chung): 85%
Macro avg (Trung bình hai lớp, không xét số lượng mẫu):
Precision: 0.80, Recall: 0.69, F1-score: 0.72
Weighted avg (Trung bình có trọng số theo số lượng mẫu mỗi lớp):
Precision: 0.84, Recall: 0.85, F1-score: 0.84

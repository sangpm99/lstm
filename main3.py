from keras.models import load_model
import numpy as np

# Tải lại mô hình từ file
model = load_model('lstm_model.h5')

# Dự đoán với dữ liệu mới
new_RF_KienGiang = [1, 2.3, 4.5]
new_RF_LeThuy = [0, 0.5, 1.0, 1.5]
new_RF_DongHoi = [3.6, 3.8, 9.4]
new_WL_KienGiang = [7.12, 7.07, 6.99]
new_WL_LeThuy = [0.57, 0.54, 0.51]
new_WL_DongHoi = [0.6, 0.75, 0.9]

# Gộp nhóm
new_data = []
new_data.append(new_RF_KienGiang + new_RF_LeThuy + new_RF_DongHoi + new_WL_KienGiang + new_WL_LeThuy + new_WL_DongHoi)

# Chuyển đổi dữ liệu mới sang mảng
new_data = np.array(new_data, dtype=float)  # shape: (1, 19)

# Reshape dữ liệu mới
new_data = np.reshape(new_data, (new_data.shape[0], new_data.shape[1], 1))  # shape: (1, 19, 1)

# Dự đoán giá trị E tiếp theo
new_predictions = model.predict(new_data)

# In kết quả dự đoán
print(new_predictions[0][0])
# Time Series Cross Validation and R-Squared (3h past)
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.models import load_model
import matplotlib.pyplot as plt

# Lấy dữ liệu từ excel
df = pd.read_csv('dataset.csv', header=1, names=['Time',
                                                 'RF_KienGiang',
                                                 'RF_LeThuy',
                                                 'RF_DongHoi',
                                                 'WL_KienGiang',
                                                 'WL_LeThuy',
                                                 'WL_DongHoi'])

# Chuyển về dạng list
RF_KienGiang = df['RF_KienGiang'].values.tolist()
RF_LeThuy = df['RF_LeThuy'].values.tolist()
RF_DongHoi = df['RF_DongHoi'].values.tolist()
WL_KienGiang = df['WL_KienGiang'].values.tolist()
WL_LeThuy = df['WL_LeThuy'].values.tolist()
WL_DongHoi = df['WL_DongHoi'].values.tolist()

list_max = max([RF_KienGiang, RF_LeThuy, RF_DongHoi, WL_KienGiang, WL_LeThuy, WL_DongHoi])
length = len(list_max)
data = []
target = []

# Thiết lập dữ liệu 3h quá khứ (6 cột)
for i in range(length - 3):
    if (
        len(RF_KienGiang) >= i + 3 and
        len(RF_LeThuy) >= i + 3 and
        len(RF_DongHoi) >= i + 3 and
        len(WL_KienGiang) >= i + 3 and
        len(WL_LeThuy) >= i + 3 and
        len(WL_DongHoi) >= i + 3
    ):
        data.append(
            RF_KienGiang[i:i + 3] +
            RF_LeThuy[i:i + 3] +
            RF_DongHoi[i:i + 3] +
            WL_KienGiang[i:i + 3] +
            WL_LeThuy[i:i + 3] +
            WL_DongHoi[i:i + 3])
        target.append(WL_LeThuy[i + 3])

# Chuyển đổi về dạng mảng numpy
data = np.array(data, dtype=float)
target = np.array(target, dtype=float)

# Đổi chiều: 2D (1, 19) => 3D (1, 19, 1)
data = np.reshape(data, (data.shape[0], data.shape[1], 1))

# Thiết lập model, khởi tạo LSTM cùng các tham số
model = Sequential()
model.add(LSTM(64, input_shape=(18, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Thiết lập Time Series với split = 5
tscv = TimeSeriesSplit(n_splits=5)
r_squared_scores = []
mse_scores = []
mae_scores = []
smse_scores = []
i = 1
predict_list = []

# Thực hiện lặp 5 split cho việc train và test
for train, test in tscv.split(data):
    X_train, X_test = data[train], data[test]
    y_train, y_test = target[train], target[test]
    model.fit(X_train, y_train, epochs=100, batch_size=512, verbose=0)
    predictions = model.predict(X_test)
    predictions = np.array(predictions, dtype=float)
    predict_round = np.round(predictions, decimals=2)   # Làm tròn 2 đơn vị sau dấu phẩy

    # Tạo danh sách dự đoán cho mỗi split (length = target/5 = 1676) (5 Split = 8380 sample)
    predict_list.append(predict_round)
    r_squared = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    smse = mse / np.var(y_test)

    # KQ các phương pháp đánh giá cho mỗi split
    r_squared_scores.append(r_squared)
    mse_scores.append(mse)
    mae_scores.append(mae)
    smse_scores.append(smse)
    print(f"Kết quả R-Squared lần {i}: {r_squared}")
    print(f"Kết quả MSE lần {i}: {mse}")
    print(f"Kết quả MAE lần {i}: {mae}")
    print(f"Kết quả SMSE lần {i}: {smse}")
    i = i + 1

# Tính trung bình
mean_r_squared = np.mean(r_squared_scores)
mean_mse = np.mean(mse_scores)
mean_mae = np.mean(mae_scores)
mean_smse = np.mean(smse_scores)
print(f"R-Squared trung bình: {mean_r_squared}")
print(f"MSE trung bình: {mean_mse}")
print(f"MAE trung bình: {mean_mae}")
print(f"SMSE trung bình: {mean_smse}")

predict_list = np.array(predict_list, dtype=float)
# Chuyển đổi danh sách dự đoán từ 3D (5, 1676, 1) => 1D (8380, )
predict_list = np.concatenate(predict_list).ravel()
# Vì 1680 sample đầu dùng để train nên chỉ lấy 8380 sample còn lại
count = len(target) - len(predict_list)  # 10060 - 8380 = 1680
real = target[count:]
# Chuẩn bị và xuất file excel để so sánh giá trị thực tế và dự đoán
dataFrame = pd.DataFrame({
    'Thuc te': real,
    'Du doan': predict_list
})
# dataFrame.to_csv('compare2.csv', index=False)

# Lưu mô hình vào file
# model.save('lstm_model2.h5')
# model = load_model('lstm_model2.h5')

plt.plot(real, color='blue', label='Thực tế')  # Đặt màu xanh cho đường x
plt.plot(predict_list, color='red', label='Dự đoán')  # Đặt màu đỏ cho đường y
plt.xlabel('Trục x')
plt.ylabel('Trục y')
plt.title('Biểu đồ so sánh kết quả dự đoán và giá trị thực tế')
plt.legend()  # Hiển thị chú thích
plt.show()

# # Dự đoán với dữ liệu mới
# new_RF_KienGiang = [1, 2.3, 4.5]
# new_RF_LeThuy = [0, 0.5, 1.0]
# new_RF_DongHoi = [3.6, 3.8, 9.4]
# new_WL_KienGiang = [7.12, 7.07, 6.99]
# new_WL_LeThuy = [0.57, 0.54, 0.51]
# new_WL_DongHoi = [0.6, 0.75, 0.9]
#
# # Gộp nhóm
# new_data = []
# new_data.append(new_RF_KienGiang + new_RF_LeThuy + new_RF_DongHoi + new_WL_KienGiang + new_WL_LeThuy + new_WL_DongHoi)
#
# # Chuyển đổi dữ liệu mới sang mảng
# new_data = np.array(new_data, dtype=float)  # shape: (1, 19)
#
# # Reshape dữ liệu mới thành
# new_data = np.reshape(new_data, (new_data.shape[0], new_data.shape[1], 1))  # shape: (1, 19, 1)
#
# # Dự đoán giá trị E tiếp theo
# new_predictions = model.predict(new_data)
#
# # In kết quả dự đoán
# print(new_predictions[0][0])
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Đường dẫn đến thư mục chứa dữ liệu cử chỉ tay
data_dir = 'path_to_your_dataset_directory'

# List các nhãn cử chỉ tay
labels = ['ngon_tro', 'ngon_giua', 'ngon_ut', 'ngon_ap_ut', 'ngon_cai']

# Khởi tạo danh sách để lưu dữ liệu và nhãn
data = []
target = []

# Đọc dữ liệu từ thư mục chứa cử chỉ tay
for label in labels:
    label_dir = os.path.join(data_dir, label)
    for filename in os.listdir(label_dir):
        image_path = os.path.join(label_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))
        data.append(image)
        target.append(label)

# Chuyển đổi dữ liệu và nhãn sang dạng numpy array
data = np.array(data)
target = np.array(target)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# Chuẩn hóa giá trị pixel về khoảng từ 0 đến 1
train_data = train_data / 255.0
test_data = test_data / 255.0

# Xây dựng mô hình CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])

# Compile và huấn luyện mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_target, epochs=10, batch_size=32, validation_data=(test_data, test_target))

# Lưu mô hình đã huấn luyện
model.save('hand_gesture_model.h5')

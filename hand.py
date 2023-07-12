import cv2
import tensorflow as tf
import numpy as np
import pyautogui

# Load mô hình đã được đào tạo
model = tf.keras.models.load_model('train_signs.h5')

# Kích thước cửa sổ video
frame_width, frame_height = 640, 480

# Khởi tạo video stream
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

# Định nghĩa các nhãn cử chỉ tay
labels = ['ngon_tro', 'ngon_giua', 'ngon_ut', 'ngon_ap_ut', 'ngon_cai']

while True:
    # Đọc từng khung hình từ video stream
    ret, frame = cap.read()

    # Lật ảnh theo chiều ngang để giống gương
    frame = cv2.flip(frame, 1)

    # Tiền xử lý ảnh đầu vào
    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0)

    # Dự đoán cử chỉ tay từ ảnh đầu vào
    prediction = model.predict(input_data)
    predicted_label = labels[np.argmax(prediction)]

    # Hiển thị kết quả dự đoán
    cv2.putText(frame, f"Gesture: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Hiển thị video
    cv2.imshow("Frame", frame)

    # Xử lý sự kiện bàn phím
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Điều khiển chuột dựa trên cử chỉ tay
    if predicted_label == 'ngon_tro':
        pyautogui.move(10, 0)  # Di chuyển chuột sang phải
    elif predicted_label == 'ngon_giua':
        pyautogui.move(-10, 0)  # Di chuyển chuột sang trái
    elif predicted_label == 'ngon_ut':
        pyautogui.move(0, 10)  # Di chuyển chuột lên trên
    elif predicted_label == 'ngon_ap_ut':
        pyautogui.move(0, -10)  # Di chuyển chuột xuống dưới
    elif predicted_label == 'ngon_cai':
        pyautogui.click()  # Nhấp chuột

# Giải phóng bộ nhớ và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()

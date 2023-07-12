import cv2
import keyboard as keyboard
import numpy as np
import pyautogui

# Ẩn cửa sổ hiển thị
cv2.namedWindow("Real-time Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real-time Object Detection", (400, 300))
cv2.moveWindow("Real-time Object Detection", -10000, -10000)

# Khởi tạo thông số cho video
output_file = "screen_capture.mp4"  # Tên tệp video xuất ra
fps = 25  # Số khung hình mỗi giây
screen_size = (1920, 1080)  # Kích thước màn hình máy tính

# Khởi tạo đối tượng ghi video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_file, fourcc, fps, screen_size)

# Đường dẫn tới tệp tin caffe model
model_path = "model/res10_300x300_ssd_iter_140000.caffemodel"
# Đường dẫn tới tệp tin prototxt
config_path = "model/deploy.prototxt.txt"

# Tạo đối tượng dnn từ caffe model và prototxt
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

while True:
    # Chụp màn hình
    screenshot = pyautogui.screenshot()

    # Chuyển đổi ảnh chụp thành mảng numpy
    screenshot_np = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Ghi khung hình vào video
    video_writer.write(screenshot_np)

    # Tạo blob từ hình ảnh để đưa vào mạng
    blob = cv2.dnn.blobFromImage(cv2.resize(screenshot_np, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Đưa blob vào mạng để nhận diện đối tượng
    net.setInput(blob)
    detections = net.forward()

    # Duyệt qua các kết quả nhận diện
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Lọc các kết quả có độ tin cậy cao
        if confidence > 0.5:
            # Tính toán tọa độ hộp giới hạn của đối tượng
            box = detections[0, 0, i, 3:7] * np.array([screenshot_np.shape[1], screenshot_np.shape[0], screenshot_np.shape[1], screenshot_np.shape[0]])
            (x, y, w, h) = box.astype("int")

            # Tính toán tọa độ giữa khuôn mặt
            face_center_x = int(x + w / 2)
            face_center_y = int(y + h / 2)

            # Di chuyển chuột đến tọa độ giữa khuôn mặt
            pyautogui.moveTo(face_center_x, face_center_y)

    # Thoát khỏi vòng lặp nếu nhấn phím 'p'
    if keyboard.is_pressed('p'):
        break

# Dừng việc ghi video
video_writer.release()

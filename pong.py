import cv2
import numpy as np
import directkeys
import time

# Đường dẫn tới tệp tin caffe model
model_path = "model/res10_300x300_ssd_iter_140000.caffemodel"
# Đường dẫn tới tệp tin prototxt
config_path = "model/deploy.prototxt.txt"

# Tạo đối tượng dnn từ caffe model và prototxt
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Khởi tạo đối tượng VideoCapture để kết nối với camera
cap = cv2.VideoCapture(0)

# Các hằng số để điều khiển thanh trong trò chơi
PADDLE_SPEED = 0.3  # Tốc độ di chuyển của thanh
MOVE_LEFT = 'left'
MOVE_RIGHT = 'right'

# Kích thước màn hình trò chơi Pong
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

# Kích thước thanh trong trò chơi Pong
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20

# Vị trí ban đầu của thanh
paddle_x = (SCREEN_WIDTH - PADDLE_WIDTH) // 2

while True:
    # Đọc từng khung hình từ camera
    ret, frame = cap.read()

    # Tạo blob từ khung hình để đưa vào mạng
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Đưa blob vào mạng để nhận diện khuôn mặt
    net.setInput(blob)
    detections = net.forward()

    # Duyệt qua các kết quả nhận diện
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Lọc các kết quả có độ tin cậy cao
        if confidence > 0.5:
            # Tính toán tọa độ hộp giới hạn của khuôn mặt
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype("int")

            # Tính toán vị trí x của thanh dựa trên vị trí của khuôn mặt
            face_center_x = (x + w) // 2
            paddle_x = int((face_center_x / frame.shape[1]) * SCREEN_WIDTH)

    # Di chuyển thanh theo vị trí đã tính toán
    if paddle_x < SCREEN_WIDTH // 2:
        directkeys.PressKey(directkeys.PressKey("w"))
        directkeys.ReleaseKey(directkeys.ReleaseKey("w"))
    else:
        directkeys.ReleaseKey(directkeys.ReleaseKey("w"))
        directkeys.PressKey(directkeys.PressKey("w"))

    # Hiển thị khung hình kết quả
    cv2.imshow("Face Detection", frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng kết nối camera
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import pyautogui

# Đường dẫn tới tệp tin caffe model
model_path = "model/res10_300x300_ssd_iter_140000.caffemodel"
# Đường dẫn tới tệp tin prototxt
config_path = "model/deploy.prototxt.txt"

# Tạo đối tượng dnn từ caffe model và prototxt
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Khởi tạo đối tượng VideoCapture để kết nối với camera
cap = cv2.VideoCapture(0)

# Lấy kích thước màn hình
# screen_size = (int(pyautogui.size().width), int(pyautogui.size().height))

# Tạo cửa sổ hiển thị fullscreen
# cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Face Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

            # Vẽ hình chữ nhật xung quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

            # Tính toán tọa độ giữa khuôn mặt
            face_center_x = int(x + w)
            face_center_y = int(y + h / 2)

            # Di chuyển chuột đến tọa độ giữa khuôn mặt
            pyautogui.moveTo(face_center_x, face_center_y)

    # Hiển thị khung hình kết quả fullscreen
    # cv2.imshow("Face Detection", cv2.resize(frame, screen_size))
    cv2.imshow("Face Detection", frame)
    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng kết nối camera
cap.release()
cv2.destroyAllWindows()

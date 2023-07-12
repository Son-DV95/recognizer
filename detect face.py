import cv2
import numpy as np

# Đường dẫn tới tệp tin caffe model
model_path = "model/res10_300x300_ssd_iter_140000.caffemodel"
# Đường dẫn tới tệp tin prototxt
config_path = "model/deploy.prototxt.txt"

# Tạo đối tượng dnn từ caffe model và prototxt
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Khởi tạo đối tượng VideoCapture để kết nối với camera
cap = cv2.VideoCapture(0)

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

            # Tạo một ROI (Region of Interest) chỉ chứa khuôn mặt
            face_roi: any = frame[y:h, x:w]

            # Chuyển đổi ảnh khuôn mặt sang ảnh đen trắng
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Phát hiện các cạnh của khuôn mặt
            edges = cv2.Canny(face_gray, 30, 150)

            # Tìm các đường viền của khuôn mặt
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Vẽ các đường viền lên khung hình gốc
            cv2.drawContours(face_roi, contours, -1, (0, 255, 0), 2)

            # Loại bỏ nền phía sau khuôn mặt bằng cách thiết lập giá trị pixel của khuôn mặt thành màu đen
            # face_roi[edges != 0] = [0, 0, 0]

    # Hiển thị khung hình kết quả
    cv2.imshow("Face Detection", frame)

    # cv2.imshow("Face Edges", face_roi)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng kết nối camera
cap.release()
cv2.destroyAllWindows()

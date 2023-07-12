import cv2
import pyautogui

# Đường dẫn tới tệp tin dữ liệu cascade cho việc nhận diện khuôn mặt
cascade_path = "opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"

# Tạo đối tượng CascadeClassifier với tệp tin cascade
face_cascade = cv2.CascadeClassifier(cascade_path)

# Khởi tạo đối tượng VideoCapture để kết nối với camera
cap = cv2.VideoCapture(0)

# Định dạng font chữ cho việc hiển thị tên người phát hiện
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)  # Màu xanh lá cây
line_thickness = 2

# Tạo danh sách tên người phát hiện
names = ["Person 1", "Person 2", "Person 3"]  # Thay đổi danh sách theo nhu cầu

# Tính toán kích thước màn hình
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Tốc độ di chuyển của chuột
MOUSE_SPEED = 10

while True:
    # Đọc từng khung hình từ camera
    ret, frame = cap.read()

    # Chuyển đổi khung hình sang grayscale để tăng tốc độ xử lý
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong khung hình
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Duyệt qua các khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Tính toán vị trí chuột dựa trên vị trí khuôn mặt và kích thước màn hình
        mouse_x = int(x + w / 2) * SCREEN_WIDTH / frame.shape[1]
        mouse_y = int(y + h / 2) * SCREEN_HEIGHT / frame.shape[0]

        # Lật ngược hướng di chuyển của chuột
        mouse_x = SCREEN_WIDTH - mouse_x
        mouse_y = SCREEN_HEIGHT - mouse_y

        # Tăng tốc độ di chuyển của chuột
        mouse_x *= MOUSE_SPEED
        mouse_y *= MOUSE_SPEED

        # Di chuyển chuột đến vị trí khuôn mặt
        pyautogui.moveTo(mouse_x, mouse_y)

    # Hiển thị khung hình kết quả
    cv2.imshow("Face Detection", frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng kết nối camera
cap.release()
cv2.destroyAllWindows()

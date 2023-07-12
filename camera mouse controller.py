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

while True:
    # Đọc từng khung hình từ camera
    ret, frame = cap.read()

    # Chuyển đổi khung hình sang grayscale để tăng tốc độ xử lý
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong khung hình
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Điều chỉnh kích thước khung hình
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

    # Duyệt qua các khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Xác định tên người phát hiện dựa trên chỉ số khuôn mặt
        # if len(names) > 0:
        #     name = names[0]  # Lấy tên của người đầu tiên trong danh sách
        #     names = names[1:]  # Xóa tên của người đầu tiên để xử lý người tiếp theo
        #     # Hiển thị tên người phát hiện trên khung hình
        #     cv2.putText(frame, name, (x, y - 10), font, font_scale, font_color, line_thickness, cv2.LINE_AA)
        for (x, y, w, h) in faces:
            # Tính toán tọa độ của chuột dựa trên vị trí khuôn mặt
            mouse_x = int(x + w / 2)
            mouse_y = int(y + h / 2)

            # Giới hạn tọa độ chuột trong kích thước màn hình
            mouse_x = max(0, min(mouse_x, SCREEN_WIDTH))
            mouse_y = max(0, min(mouse_y, SCREEN_HEIGHT))

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

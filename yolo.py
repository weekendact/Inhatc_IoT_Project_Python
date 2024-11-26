import cv2
import requests
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO(r"yolov8s.pt")

# 라즈베리파이 서버 주소
RASPBERRY_PI_URL = "http://192.168.101.101:5000"  # 라즈베리파이 IP 주소

# 웹캠 열기
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("웹캠에서 프레임을 가져올 수 없습니다.")
            break

        # YOLOv8 객체 탐지
        results = model.predict(source=frame, conf=0.5, show=False)

        # "person" 클래스 판별
        person_detected = False
        for result in results[0].boxes.data:
            cls = int(result[5])  # 클래스 ID
            if model.names[cls] == "person":
                person_detected = True
                break

        # 라즈베리파이에 신호 전송
        if person_detected:
            print("사람이 감지되었습니다. 신호 전송...")
            try:
                requests.post(f"{RASPBERRY_PI_URL}/action", json={"status": "detected"})
            except Exception as e:
                print(f"라즈베리파이에 신호를 보낼 수 없습니다: {e}")

        # 탐지 결과 시각화
        result_frame = results[0].plot()
        cv2.imshow("Object Detection", result_frame)

        # 'q'를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

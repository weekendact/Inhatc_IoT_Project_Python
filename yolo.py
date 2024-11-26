import socket
import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO(r"yolov8s.pt")

# UDP 소켓 설정
HOST = "0.0.0.0"  # 모든 IP에서 수신
PORT = 9999
BUFFER_SIZE = 65536  # 최대 패킷 크기

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))

print(f"UDP 서버 대기 중... (포트 {PORT})")

while True:
    # 데이터 수신
    data, addr = server_socket.recvfrom(BUFFER_SIZE)
    print(f"{addr}에서 데이터 수신 ({len(data)} bytes)")

    # 데이터를 디코딩하여 이미지로 변환
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        print("프레임 디코딩 실패")
        continue

    # YOLOv8로 객체 탐지
    results = model.predict(source=frame, conf=0.5, show=False)

    # "person" 클래스 판별
    person_detected = False
    for result in results[0].boxes.data:
        cls = int(result[5])  # 클래스 ID
        if model.names[cls] == "person":
            person_detected = True
            break

    # 결과 출력
    if person_detected:
        print(1)  # 터미널에 1 출력
    else:
        print(0)  # 터미널에 0 출력 (필요시)

    # (선택) 탐지 결과 화면에 출력
    result_frame = results[0].plot()
    cv2.imshow("Object Detection", result_frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

server_socket.close()
cv2.destroyAllWindows()

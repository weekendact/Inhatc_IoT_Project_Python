import cv2
import socket
import struct
import pickle
from ultralytics import YOLO

# 서버 설정
HOST = "0.0.0.0"  # 모든 IP에서 수신
PORT = 9999

# YOLOv8 모델 로드
model = YOLO(r"yolov8s.pt")  # YOLOv8 small 모델

# 소켓 설정
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)
print("서버 대기 중...")

conn, addr = server_socket.accept()
print(f"클라이언트 연결됨: {addr}")

try:
    while True:
        # 데이터 길이 수신
        packed_size = conn.recv(4)
        if not packed_size:
            break

        data_size = struct.unpack(">L", packed_size)[0]
        data = b""

        # 데이터 수신
        while len(data) < data_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        # 프레임 복원
        frame = pickle.loads(data)

        # YOLOv8로 객체 탐지 수행
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
            print("사람이 감지되었습니다!")
        else:
            print("사람이 없습니다.")

        # 탐지 결과 시각화
        result_frame = results[0].plot()
        cv2.imshow("Object Detection", result_frame)

        # 'q'를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()

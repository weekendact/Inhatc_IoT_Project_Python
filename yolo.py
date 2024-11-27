import cv2
import requests
from ultralytics import YOLO
import streamlit as st

# 라즈베리파이 서버 주소
RASPBERRY_PI_URL = "http://192.168.101.101:5000"

# YOLO 모델 로드
car_model = YOLO("yolov8m.pt")  # COCO 데이터셋 사전 학습 모델 (car 포함)
fire_model = YOLO("best.pt")  # 화재 탐지 커스텀 모델

# Streamlit 페이지 설정
st.set_page_config(layout="wide")
st.title("YOLOv8 Car and Fire Detection")

# 웹캠 연결
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    st.error("웹캠을 열 수 없습니다.")
    st.stop()

# Streamlit 인터페이스 구성
FRAME_WINDOW = st.image([])
status_text = st.empty()  # 화재 상태 메시지 출력

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("웹캠에서 프레임을 가져올 수 없습니다.")
            break

        # 차량 탐지
        car_results = car_model.predict(source=frame, conf=0.5, show=False)
        car_frame = car_results[0].plot()  # Bounding Box 그리기

        # 화재 탐지
        fire_results = fire_model.predict(source=frame, conf=0.5, show=False)
        fire_detected = any(fire_model.names[int(result[5])] == "fire" for result in fire_results[0].boxes.data)

        # 라즈베리파이에 신호 전송 (화재가 감지된 경우)
        if fire_detected:
            status_text.write("🔴 **화재 감지: 라즈베리파이에 신호 전송 중...**")
            try:
                requests.post(f"{RASPBERRY_PI_URL}/action", json={"status": "detected"})
            except Exception as e:
                st.error(f"라즈베리파이에 신호를 보낼 수 없습니다: {e}")
        else:
            status_text.write("🟢 **화재가 감지되지 않았습니다.**")

        # 두 결과를 합성하여 화면에 표시
        fire_frame = fire_results[0].plot()
        combined_frame = cv2.addWeighted(car_frame, 0.5, fire_frame, 0.5, 0)
        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

        # Streamlit에 결과 표시
        FRAME_WINDOW.image(combined_frame, channels="RGB")

finally:
    camera.release()
    st.write("카메라가 정상적으로 종료되었습니다.")

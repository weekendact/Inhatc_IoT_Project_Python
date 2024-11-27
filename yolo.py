import cv2
import requests
import streamlit as st
from ultralytics import YOLO

# 라즈베리파이 서버 주소
RASPBERRY_PI_URL = "http://192.168.101.101:5000"

# YOLOv8 모델 로드
model = YOLO(r"fire-detection.pt")  # 화재 탐지 모델 경로

# Streamlit 페이지 설정
st.set_page_config(layout="wide")
st.title("YOLOv8 Fire Detection with Raspberry Pi Integration")

# 웹캠 선택
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    st.error("웹캠을 열 수 없습니다.")
    st.stop()

# Streamlit 인터페이스 구성
FRAME_WINDOW = st.image([])
status_text = st.empty()  # 상태 메시지 출력

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("웹캠에서 프레임을 가져올 수 없습니다.")
            break

        # YOLOv8 객체 탐지
        results = model.predict(source=frame, conf=0.5, show=False)

        # Bounding Box가 그려진 결과 프레임
        result_frame = results[0].plot()

        # "fire" 클래스 판별
        fire_detected = False
        for result in results[0].boxes.data:
            cls = int(result[5])  # 클래스 ID
            if model.names[cls] == "fire":
                fire_detected = True
                break

        # 라즈베리파이에 신호 전송
        if fire_detected:
            status_text.write("🔴 **불이 감지되었습니다! 라즈베리파이에 신호 전송 중...**")
            try:
                requests.post(f"{RASPBERRY_PI_URL}/action", json={"status": "fire_detected"})
            except Exception as e:
                st.error(f"라즈베리파이에 신호를 보낼 수 없습니다: {e}")
        else:
            status_text.write("🟢 **불이 감지되지 않았습니다.**")
            try:
                requests.post(f"{RASPBERRY_PI_URL}/action", json={"status": "no_fire"})
            except Exception as e:
                st.error(f"라즈베리파이에 신호를 보낼 수 없습니다: {e}")

        # OpenCV 이미지를 Streamlit용으로 변환
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(result_frame, channels="RGB")

finally:
    camera.release()
    st.write("카메라가 정상적으로 종료되었습니다.")

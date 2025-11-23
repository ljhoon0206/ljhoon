import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ---------------- Mediapipe 초기화 ----------------
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FACE_DETECTOR = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.4)
HAND_DETECTOR = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# ---------------- 웹 앱 전역 변수 설정 ----------------
TARGET_A_MIN, TARGET_A_MAX = 43, 47
TARGET_B_MIN, TARGET_B_MAX = 12, 15
COUNTDOWN_TIME = 3.0  # 카운트다운 시간 설정


# ---------------- Victory 제스처 판단 ----------------
def is_victory(lms, w, h):
    """검지+중지 펴짐, 약지+새끼 접힘이면 V 사인 True"""

    def c(i):
        lm = lms.landmark[i]
        return int(lm.x * w), int(lm.y * h)

    i_tip = c(8)
    m_tip = c(12)
    r_tip = c(16)
    p_tip = c(20)

    i_kn = c(5)
    m_kn = c(9)
    r_kn = c(13)
    p_kn = c(17)

    return i_tip[1] < i_kn[1] and m_tip[1] < m_kn[1] and r_tip[1] > r_kn[1] and p_tip[1] > p_kn[1]


# ---------------- 비율 계산 함수 ----------------
def get_face_distances(detection):
    keypoints = detection.location_data.relative_keypoints
    bbox_h = detection.location_data.relative_bounding_box.height

    if bbox_h == 0:
        return {'eye_mouth_ratio': 0.0, 'nose_mouth_ratio': 0.0}

    y_eye_r = keypoints[1].y
    y_eye_l = keypoints[0].y
    y_eye_center = (y_eye_r + y_eye_l) / 2
    y_mouth = keypoints[3].y
    y_nose = keypoints[2].y

    distance_eye_mouth_norm = abs(y_mouth - y_eye_center)
    eye_mouth_ratio = distance_eye_mouth_norm / bbox_h

    distance_nose_mouth_norm = abs(y_mouth - y_nose)
    nose_mouth_ratio = distance_nose_mouth_norm / bbox_h

    return {
        'eye_mouth_ratio': eye_mouth_ratio,
        'nose_mouth_ratio': nose_mouth_ratio
    }


# ---------------- 게이지 그리기 함수 ----------------
def draw_gauge(img, ratio_percent, x_offset, target_min, target_max, label):
    """화면 왼쪽에 수직 게이지를 그립니다."""
    gauge_x, gauge_y = 50 + x_offset, 80
    gauge_w, gauge_h = 20, 200

    ratio_percent_clamped = max(0, min(100, ratio_percent))

    is_target = target_min <= ratio_percent_clamped <= target_max

    target_color = (0, 255, 0)
    base_color = (255, 255, 255)
    fill_color = target_color if is_target else (0, 0, 255)

    # 게이지 배경 (테두리)
    cv2.rectangle(img, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), base_color, 2)

    # 게이지 채우기
    fill_height = int(gauge_h * (ratio_percent_clamped / 100))
    fill_y_start = gauge_y + gauge_h - fill_height
    cv2.rectangle(img, (gauge_x, fill_y_start), (gauge_x + gauge_w, gauge_y + gauge_h), fill_color, cv2.FILLED)

    # 타겟 영역 표시
    y_min = gauge_y + gauge_h - int(gauge_h * (target_min / 100))
    y_max = gauge_y + gauge_h - int(gauge_h * (target_max / 100))

    cv2.line(img, (gauge_x - 5, y_min), (gauge_x + gauge_w + 5, y_min), (0, 255, 255), 1)
    cv2.line(img, (gauge_x - 5, y_max), (gauge_x + gauge_w + 5, y_max), (0, 255, 255), 1)

    cv2.putText(img, label, (gauge_x - 10, gauge_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, base_color, 1)
    cv2.putText(img, f"{ratio_percent_clamped}%", (gauge_x - 10, gauge_y + gauge_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_color, 2)

    return is_target, ratio_percent_clamped


# ---------------- VideoProcessor 클래스 (핵심) ----------------
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        # 캡처 상태
        self.captured = False
        self.last_capture_time = 0

        # 카운트다운 상태
        self.countdown_active = False
        self.countdown_start_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_h, img_w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---------------- 1. 얼굴 인식 및 비율 확인 ----------------
        face_res = FACE_DETECTOR.process(rgb)
        face_detected = False
        ratio_ok_A, ratio_ok_B = False, False
        ratio_A_percent, ratio_B_percent = 0, 0

        if face_res.detections:
            face_detected = True
            d = face_res.detections[0]

            current_ratios = get_face_distances(d)

            ratio_A_percent = int(current_ratios['eye_mouth_ratio'] * 100)
            ratio_B_percent = int(current_ratios['nose_mouth_ratio'] * 100)

            ratio_ok_A = TARGET_A_MIN <= ratio_A_percent <= TARGET_A_MAX
            ratio_ok_B = TARGET_B_MIN <= ratio_B_percent <= TARGET_B_MAX

        # ---------------- 2. 손 인식 및 V 사인 확인 ----------------
        hand_res = HAND_DETECTOR.process(rgb)
        victory_detected = False

        if hand_res.multi_hand_landmarks:
            for handLms in hand_res.multi_hand_landmarks:
                if is_victory(handLms, img_w, img_h):
                    victory_detected = True
                    cv2.putText(img, "VICTORY!", (50, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    break

        # ---------------- 3. 게이지 표시 ----------------
        draw_gauge(img, ratio_A_percent, 0, TARGET_A_MIN, TARGET_A_MAX, "E-M Ratio")
        draw_gauge(img, ratio_B_percent, 70, TARGET_B_MIN, TARGET_B_MAX, "N-M Ratio")

        total_ratio_ok = ratio_ok_A and ratio_ok_B

        all_conditions_met = face_detected and victory_detected and total_ratio_ok

        # ---------------- 4. 카운트다운 및 캡처 로직 (세션 상태 복원) ----------------
        if all_conditions_met:
            # A. 모든 조건 충족 & 캡처 대기 상태 -> 카운트다운 시작
            if not self.captured and not self.countdown_active:
                self.countdown_active = True
                self.countdown_start_time = time.time()
                st.session_state.capture_message = f"카운트다운 시작! {COUNTDOWN_TIME}초 유지하세요."

            # B. 카운트다운 진행 중
            if self.countdown_active:
                elapsed = time.time() - self.countdown_start_time
                countdown_value = COUNTDOWN_TIME - elapsed

                # 카운트다운 텍스트 표시
                countdown_display = max(0, int(countdown_value) + 1)

                cv2.putText(img, f"Capturing in: {countdown_display}", (img_w // 2 - 150, img_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

                # C. 카운트다운 종료 -> 캡처 실행
                if countdown_value <= 0:
                    self.countdown_active = False
                    self.captured = True
                    self.last_capture_time = time.time()

                    # ⭐⭐ 핵심: set_result 대신 st.session_state에 직접 저장하고 rerun 호출 복원 ⭐⭐
                    # (Legacy API 환경에서 캡처 UI 업데이트를 위한 유일한 방법일 수 있음)
                    st.session_state.captured_image_bytes = cv2.imencode('.png', img)[1].tobytes()
                    st.session_state.capture_ready = True
                    st.session_state.capture_message = "✅ 촬영 성공! 아래에서 다운로드하세요."
                    st.rerun()  # UI 업데이트 강제 요청

        else:
            # 조건 불충족 시 카운트다운 중단
            if self.countdown_active:
                self.countdown_active = False
                st.session_state.capture_message = "⏳ 조건 미달로 카운트다운 중단."

        # ---------------- 5. 캡처 이미지 유지 및 리셋 ----------------
        if self.captured:
            if time.time() - self.last_capture_time > 3.0:
                self.captured = False
                st.session_state.capture_message = "⏳ 다시 촬영 준비 완료."

        # ---------------- 6. 상태 표시 ----------------
        status_text = (
            f"Face: {face_detected} | V: {victory_detected} | "
            f"Ratio A(E-M): {ratio_ok_A} | Ratio B(N-M): {ratio_ok_B}"
        )
        cv2.putText(img, status_text,
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- Streamlit 메인 함수 ----------------
def main():
    st.set_page_config(page_title="비율 & V 사인 검출기", layout="wide")

    st.title("📸 비율 최적화 V-사인 자동 캡처 웹 앱")
    st.markdown("""
        모든 조건이 충족되면 **3초 카운트다운** 후 자동으로 캡처됩니다. 3초 동안 자세를 유지하세요!
    """)
    st.markdown("---")

    # Session State 초기화
    if 'capture_ready' not in st.session_state:
        st.session_state.capture_ready = False
        st.session_state.captured_image_bytes = None
        st.session_state.capture_message = "카메라를 켜고 자세를 잡아주세요."

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("실시간 웹캠 스트림 (비전 처리)")
        webrtc_ctx = webrtc_streamer(
            key="media-pipe-detector",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.info(f"현재 상태: **{st.session_state.capture_message}**")

    with col2:
        st.subheader("✅ 캡처 결과 및 다운로드")

        st.markdown(
            f"""
            **✅ 캡처 조건 (모두 충족해야 함):**
            * 얼굴 감지 (Face Detected)
            * V-사인 감지 (Victory Gesture)
            * **눈-입 비율:** ${TARGET_A_MIN}\% \sim {TARGET_A_MAX}\%$
            * **코-입 비율:** ${TARGET_B_MIN}\% \sim {TARGET_B_MAX}\%$
            """
        )

        # ⭐⭐ 핵심: webrtc_ctx.video_processor_result 확인 로직 제거 ⭐⭐
        # (Legacy API 환경에서는 이 속성이 없으므로)

        # 캡처된 이미지가 있을 때만 표시 및 다운로드 버튼 활성화
        if st.session_state.capture_ready and 'captured_image_bytes' in st.session_state and st.session_state.captured_image_bytes is not None:
            st.image(st.session_state.captured_image_bytes, caption="최근 캡처 이미지", use_column_width=True)

            # 다운로드 버튼
            st.download_button(
                label="🖼️ 캡처 이미지 다운로드",
                data=st.session_state.captured_image_bytes,
                file_name=f"capture_optimal_{int(time.time())}.png",
                mime="image/png"
            )
        elif st.session_state.capture_ready == False:
            st.warning("아직 캡처된 이미지가 없습니다. 조건을 충족시켜보세요!")


if __name__ == "__main__":
    main()
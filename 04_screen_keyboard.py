import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "Backspace"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", "Enter"],
        ["Shift", "Z", "X", "C", "V", "B", "N", "M", "Space"]]

# ตั้งค่าดีเลย์ของ PyAutoGUI
pyautogui.PAUSE = 0.1
shift_active = False

# กำหนดตำแหน่งและขนาดปุ่ม
KEYBOARD_X = 200  # ตำแหน่ง X (ซ้าย-ขวา)
KEYBOARD_Y = 400  # ตำแหน่ง Y (ขึ้น-ลง)
KEY_WIDTH = 80    # ความกว้างของปุ่ม (ค่าเพิ่ม = ปุ่มใหญ่ขึ้น)
KEY_HEIGHT = 100  # ความสูงของปุ่ม (ค่าเพิ่ม = ปุ่มสูงขึ้น)
SPECIAL_KEY_WIDTH = 160  # ความกว้างปุ่มพิเศษ เช่น Space, Backspace, Enter, Shift

# ฟังก์ชันวาดปุ่มคีย์บอร์ด
def draw_keyboard(img, key_list):
    key_positions = []
    y_offset = KEYBOARD_Y
    for i, row in enumerate(key_list):
        x_offset = KEYBOARD_X
        for key in row:
            width = SPECIAL_KEY_WIDTH if key in ["Space", "Backspace", "Enter", "Shift"] else KEY_WIDTH
            top_left = (x_offset, y_offset)
            bottom_right = (x_offset + width, y_offset + KEY_HEIGHT)
            cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 2)
            cv2.putText(img, key, (x_offset + 10, y_offset + int(KEY_HEIGHT/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            key_positions.append((key, top_left, bottom_right))
            x_offset += width + 10
        y_offset += KEY_HEIGHT + 10
    return key_positions

def check_key_press(finger_pos, key_positions):
    x, y = finger_pos
    for key, top_left, bottom_right in key_positions:
        if top_left[0] < x < bottom_right[0] and top_left[1] < y < bottom_right[1]:
            return key
    return None

# ตรวจจับว่ามีการกดนิ้วโป้งหรือไม่
def is_thumb_pressed(index_finger, thumb, threshold=50):
    """ ตรวจจับว่าระยะระหว่างนิ้วชี้และนิ้วโป้งเข้าใกล้กันหรือไม่ """
    distance = np.linalg.norm(np.array(index_finger) - np.array(thumb))
    return distance < threshold

# เปิดกล้องและตั้งค่าความละเอียดเป็น 720p
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # ความกว้าง
cap.set(4, 720)   # ความสูง

last_key = None
last_press_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # พลิกภาพให้เหมือนกระจก
    frame = cv2.resize(frame, (1280, 720))  # ปรับขนาด
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # วาดคีย์บอร์ด
    key_positions = draw_keyboard(frame, keys)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ดึงตำแหน่งปลายนิ้วชี้ (Landmark 8) และนิ้วโป้ง (Landmark 4)
            h, w, _ = frame.shape
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            x_index, y_index = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # วาดจุดที่นิ้วชี้และนิ้วโป้ง
            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x_thumb, y_thumb), 10, (0, 0, 255), -1)

            pressed_key = check_key_press((x_index, y_index), key_positions)

            if pressed_key and is_thumb_pressed((x_index, y_index), (x_thumb, y_thumb)):
                if pressed_key != last_key or time.time() - last_press_time > 0.5:
                    print(f"Pressed: {pressed_key}")  # แสดงผลบน Terminal
                    last_key = pressed_key
                    last_press_time = time.time()
                    if pressed_key == "Space":
                        pyautogui.press("space")
                    elif pressed_key == "Enter":
                        pyautogui.press("enter")
                    elif pressed_key == "Backspace":
                        pyautogui.press("backspace")
                    elif pressed_key == "Shift":
                        shift_active = not shift_active  # สลับโหมด Shift
                    else:
                        if shift_active:
                            pyautogui.press(pressed_key.upper())
                        else:
                            pyautogui.press(pressed_key.lower())

                    cv2.rectangle(frame, (x_index-20, y_index-20), (x_index+20, y_index+20), (0, 255, 255), -1)

    cv2.imshow("Virtual Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

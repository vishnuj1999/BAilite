import cv2
import math
import time
import requests
from ultralytics import YOLO

# Block 1: Declarations
def declare_variables():
    global blue, red, green, yellow, cyan, magenta, black, white
    global intervel, D90_turn_delay

    blue = (255, 0, 0)
    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)
    magenta = (255, 0, 255)
    black = (0, 0, 0)
    white = (255, 255, 255)

    intervel = 0.01
    D90_turn_delay = 500

# Block 2: Bot Control Functions
def bot_control_functions():
    def delay(sec):
        time.sleep(sec)

    def bot_move(com):
        try:
            response = requests.get(f"http://{host}/?cmd={com}")
            if response.status_code == 200:
                print(f"bot_move({com})")
            else:
                print(f"Failed to send command. HTTP Status Code: {response.status_code}")
            delay(intervel)
        except requests.exceptions.RequestException as err:
            print(f"Request Exception - Move: {err}")

    def bot_speed(l_pwm, r_pwm):
        print(f"bot_speed(L : {l_pwm} | R : {r_pwm})")
        try:
            requests.get(f"http://{host}/?cmd=setspeed={l_pwm, r_pwm}")
        except requests.exceptions.RequestException as err:
            print(f"Request Exception - Speed: {err}")

    def stop():
        bot_move("s")
        print("stop")

    def sign_detection(classname):
        if classname == "go":
            bot_move("f(500)")
        elif classname == "left":
            bot_move(f"l({D90_turn_delay})")
        elif classname == "right":
            bot_move(f"r({D90_turn_delay})")
        elif classname == "stop":
            bot_move("s")

    return delay, bot_move, bot_speed, stop, sign_detection

# Block 3: Initialize Camera
def init_camera():
    return cv2.VideoCapture(0)

# Block 4: Host & Model Configuration
def init_host_and_model():
    host = "192.168.20.10"
    model = YOLO("Models/turn-signs-v4_640_yolov8_epo_100.pt")
    confidence = 0.80
    class_names = ['go', 'left', 'right', 'stop']
    return host, model, confidence, class_names

# Block 5: Calculate FPS
def update_fps(fps_data):
    frame_counter, start_time = fps_data
    frame_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        fps = frame_counter / elapsed_time
        frame_counter = 0
        start_time = time.time()
    else:
        fps = 0
    return fps, frame_counter, start_time

# Block 6: Perform Detection & Drawing
def detect_objects(model, img, confidence_score, class_names, obj_count, sign_detection_func):
    results = model(source=img, conf=confidence_score)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            obj_count += 1
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cv2.rectangle(img, (x1, y1), (x2, y2), green, 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            conf_str = str(int(confidence * 100))

            cls = int(box.cls[0])
            width = x2 - x1
            height = y2 - y1
            print(f"{class_names[cls]} Size : {width} x {height}")

            print(f"m_count : {obj_count}")
            if obj_count >= 4:
                sign_detection_func(class_names[cls])
                obj_count = 0

            text_size = f"Size: {width} x {height}"
            cv2.putText(img, text_size, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 2)

            org = (x1, y1)
            cv2.putText(img, f"{class_names[cls]}:{conf_str}%", org, cv2.FONT_HERSHEY_SIMPLEX, 1, magenta, 2)

    return img, obj_count

# Block 7: Display Image
def display_frame(img, fps):
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cyan, 2)
    cv2.imshow("Bot Webcam", img)

# Block 8: Main Loop
def main_loop():
    declare_variables()
    global host
    host, model, confidence_score, class_names = init_host_and_model()
    cap = init_camera()

    delay, bot_move, bot_speed, stop, sign_detection_func = bot_control_functions()

    frame_counter = 0
    start_time = time.time()
    obj_count = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        fps, frame_counter, start_time = update_fps((frame_counter, start_time))
        img, obj_count = detect_objects(model, img, confidence_score, class_names, obj_count, sign_detection_func)
        display_frame(img, fps)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit 'q' was pressed.")
            stop()
            break

    cap.release()
    cv2.destroyAllWindows()

# Block 9: Run Main
if __name__ == "__main__":
    main_loop()

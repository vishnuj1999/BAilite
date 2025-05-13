import cv2
import easyocr
import requests

# Block 1: Declare Globals
def declare_globals():
    global reader, cap, host_bot
    reader = easyocr.Reader(['en'], gpu=False)
    bot_num = input("Enter bot number: ")
    host_bot = f"192.168.{bot_num}.10"
    cap = cv2.VideoCapture(0)

# Block 2: Send Command to Bot
def send_string_to_ip(data_to_send):
    try:
        endpoint_url = f"http://{host_bot}/?cmd={data_to_send}"
        response = requests.get(endpoint_url)
        print(f"Sent: {data_to_send} | Response: {response.status_code}")
    except Exception as e:
        print(f"Error sending command: {e}")

# Block 3: Capture Frame from Camera
def capture_frame():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from camera.")
        return None
    return frame

# Block 4: Perform OCR on Frame
def perform_ocr(frame):
    results = reader.readtext(frame)
    if results:
        return results[0][1]
    else:
        return "No text detected"

# Block 5: Decide Movement Based on Text
def decide_movement(text):
    text_lower = text.lower()
    if all(char in text_lower for char in ["s", "t", "o", "p"]):
        send_string_to_ip("s")
    elif all(char in text_lower for char in ["g", "o"]):
        send_string_to_ip("f")
    elif all(char in text_lower for char in ["r", "i", "g", "h", "t"]):
        send_string_to_ip("r")
    elif all(char in text_lower for char in ["l", "e", "f", "t"]):
        send_string_to_ip("l")
    else:
        send_string_to_ip("s")  # Default to stop on unrecognized text

# Block 6: Annotate Frame
def annotate_frame(frame, text):
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Block 7: Display Frame
def display_frame(frame):
    cv2.imshow('OCR', frame)

# Block 8: Main Loop
def main_loop():
    while True:
        frame = capture_frame()
        if frame is None:
            break

        text = perform_ocr(frame)
        decide_movement(text)
        annotated_frame = annotate_frame(frame, text)
        display_frame(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key pressed.")
            break

# Block 9: Cleanup
def cleanup():
    cap.release()
    cv2.destroyAllWindows()

# Entrypoint
if __name__ == "__main__":
    declare_globals()
    main_loop()
    cleanup()

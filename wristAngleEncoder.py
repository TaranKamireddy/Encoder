import cv2
import mediapipe as mp
import numpy as np
from random import randint

# from ml import predict_y, main

import serial
import time

# === CONFIG ===
COM_PORT = 'COM3'
BAUD_RATE = 1000000
OUTPUT_FILE = 'data.csv'

# === SETUP ===
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Let Arduino reset

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LANDMARKS_TO_KEEP = [
    0,  # Wrist
    # 1, 2, 3, 4 # Thumb
    5, 6, #7, 8,    # Index Finger (Metacarpal, Proximal, Intermediate, Tip)
    9, 10, #11, 12,  # Middle Finger
    13, 14, #15, 16, # Ring Finger
    17, 18, #19, 20  # Pinky
]

HAND_CONNECTIONS = [
    (0, 5), (5, 6), #(6, 7), (7, 8),   # Index Finger
    (5, 9), (9, 10), #(10, 11), (11, 12), # Middle Finger
    (9, 13), (13, 14), #(14, 15), (15, 16), # Ring Finger
    (13, 17), (0, 17), (17, 18), #(18, 19), (19, 20)  # Pinky
]

cap = cv2.VideoCapture(0)

def calculate_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle_rad)

def readEncoder(ser):
    data = ser.read(ser.in_waiting)  # Read everything in the buffer
    lines = data.split(b'\n')        # Split into lines
    
    if len(lines) > 1:
        return lines[-2]  # Last full line (before final \n)
    elif lines:
        return lines[0]
    else:
        return None


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3) as hands:
    data = []
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        height, width, _ = image.shape
        angle = 0
        # print(height, width)
        black_image = np.zeros((height, width, 3), dtype=np.uint8)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)


        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = {}
                encoder = {}
                wristPos = hand_landmarks.landmark[0]
                wristPixel = (int(wristPos.x * width), int(wristPos.y * height))
                # print(wristPixel, wristPos.z)
                indexF = hand_landmarks.landmark[5]
                r = (indexF.x-wristPos.x)**2 + (indexF.y-wristPos.y)**2
                indexTip = hand_landmarks.landmark[12].z
                angle = calculate_angle([r,0], [r,indexF.z]) * abs(indexTip)/indexTip
                line = readEncoder(ser).decode(errors='ignore').strip()
                if line[1:].isnumeric() or line.isnumeric():
                    print(angle, line)
                    data.append((line, angle))
                    
                # print(angle)
                
                
                for idx in LANDMARKS_TO_KEEP:
                    landmark = hand_landmarks.landmark[idx]
                    
                    # points[idx] = (int(landmark.x * width) - wristPos[0], int(landmark.y * height + height) - wristPos[1])
                    #print(points[idx])
                    points[idx] = (int(landmark.x * width), int(landmark.y * height))

                    
                    encoder[idx] = (randint(0, width - 1), randint(0, height - 1))


                    cv2.circle(black_image, points[idx], 5, (0, 255, 0), -1)
                    # cv2.circle(black_image, (encoder[idx][0] + width, encoder[idx][1]), 5, (0, 255, 0), -1)

                for connection in HAND_CONNECTIONS:
                    if connection[0] in points and connection[1] in points:
                        cv2.line(black_image, points[connection[0]], points[connection[1]], (255, 0, 0), 2)
                        p1 = (encoder[connection[0]][0] + width, encoder[connection[0]][1])
                        p2 = (encoder[connection[1]][0] + width, encoder[connection[1]][1])
                        # cv2.line(black_image, p1, p2, (255, 0, 0), 2)

        flip = cv2.flip(black_image, 1)

        offset = -80
        xText = width//2 + offset
        yText = 50
        # cv2.putText(flip, "Our Model", (xText, yText), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        # cv2.putText(flip, "Mediapipe", (xText + width, yText), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        # cv2.putText(flip, f"Angle: {angle:.2f}", (xText + width, yText + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(flip, "Mediapipe", (xText, yText), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(flip, f"Angle: {angle:.2f}", (xText, yText + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.imshow('Side-by-Side Hand Model', flip)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
        with open(OUTPUT_FILE, 'w', newline='') as f:
          # print(data)
          for d in data:
            f.write(f"{d[0]},{d[1]}\n")
            f.flush()

        # main()


cap.release()
cv2.destroyAllWindows()
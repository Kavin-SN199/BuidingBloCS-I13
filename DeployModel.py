# Run this code in Visual Studio Code for easier access to webcam and downloaded files

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the model from downloads
model = tf.keras.models.load_model('/Users/sathia/Downloads/asl_cnn_model.h5') # Change this to the file path of the trained model.

# Map labels
label_map = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
]                                                          
Num_Classes = model.output_shape[-1]

if len(label_map) < Num_Classes:
    label_map += [f"UNK_{i}" for i in range(len(label_map), Num_Classes)]
elif len(label_map) > Num_Classes:
    label_map = label_map[:Num_Classes]

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Gets surrounding frames of the hand's features which helps the model predict
def hand_surrounding(hand_features, image_shape):
    img_h, img_w, _ = image_shape
    x_coords = [lm.x * img_w for lm in hand_features.landmark]
    y_coords = [lm.y * img_h for lm in hand_features.landmark]
    x_min = max(0, int(min(x_coords)) - 20)
    y_min = max(0, int(min(y_coords)) - 20)
    x_max = min(img_w, int(max(x_coords)) + 20)
    y_max = min(img_h, int(max(y_coords)) + 20)
    return x_min, y_min, x_max, y_max

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    predicted_label = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_features in results.multi_hand_landmarks:
            # Draw hand features
            mp_drawing.draw_landmarks(frame, hand_features, mp_hands.HAND_CONNECTIONS)
            x_min, y_min, x_max, y_max = hand_surrounding(hand_features, frame.shape)
            hand_crop = frame[y_min:y_max, x_min:x_max]

            # Preprocess to 64x64 grayscale image
            gray_hand = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            resized_hand = cv2.resize(gray_hand, (64, 64))
            input_image = resized_hand.reshape(1, 64, 64, 1).astype(np.float32) / 255.0

            # Make prediction
            prediction = model.predict(input_image, verbose=0)

            # Get the index of the highest confidence class
            predicted_index = int(np.argmax(prediction))

            # Safeguard: Ensure predicted_index is within label_map bounds
            if predicted_index < len(label_map):
                predicted_label = label_map[predicted_index]
            else:
                predicted_label = "Unknown"

            # Get confidence score for the predicted class
            confidence = float(prediction[0][predicted_index])

            confidence_level = 0.65
            if confidence > confidence_level:
                predicted_label = label_map[predicted_index]
            else:
                predicted_label = "Unsure"

            # Format output string
            confidence_text = f"{confidence:.2f}"

            # Show the translated letter next to the hand
            text = f"{predicted_label} ({confidence:.2f})"
            cv2.putText(frame, text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Image-Based Classification", frame)

    # Press 'q' to exit webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean
cap.release()
cv2.destroyAllWindows()
hands.close()

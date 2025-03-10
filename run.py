from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
from math import sqrt
import time

app = Flask(__name__)

COUNTER = 0
TOTAL_BLINKS = 0
start_time = None

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Landmarks for eyes from mesh_map.jpg
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_capture = None
video_feed_active = False

def landmarksDetection(image, results, draw=False):
    image_height, image_width = image.shape[:2]
    mesh_coordinates = [(int(point.x * image_width), int(point.y * image_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        for i in LEFT_EYE + RIGHT_EYE:
            cv2.circle(image, mesh_coordinates[i], 2, (0, 255, 0), -1)
    return mesh_coordinates

def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

def blinkRatio(image, landmarks, right_indices, left_indices):
    right_eye_landmark1 = landmarks[right_indices[0]]
    right_eye_landmark2 = landmarks[right_indices[8]]
    right_eye_landmark3 = landmarks[right_indices[12]]
    right_eye_landmark4 = landmarks[right_indices[4]]

    left_eye_landmark1 = landmarks[left_indices[0]]
    left_eye_landmark2 = landmarks[left_indices[8]]
    left_eye_landmark3 = landmarks[left_indices[12]]
    left_eye_landmark4 = landmarks[left_indices[4]]

    right_eye_horizontal_distance = euclideanDistance(right_eye_landmark1, right_eye_landmark2)
    right_eye_vertical_distance = euclideanDistance(right_eye_landmark3, right_eye_landmark4)
    left_eye_vertical_distance = euclideanDistance(left_eye_landmark3, left_eye_landmark4)
    left_eye_horizontal_distance = euclideanDistance(left_eye_landmark1, left_eye_landmark2)

    # Calculate ratios
    right_eye_ratio = right_eye_horizontal_distance / right_eye_vertical_distance
    left_eye_ratio = left_eye_horizontal_distance / left_eye_vertical_distance
    eyes_ratio = (right_eye_ratio + left_eye_ratio) / 2

    # Adjust thresholds based on testing and observations
    if right_eye_ratio > 3.0 and left_eye_ratio > 3.0:
        return eyes_ratio
    else:
        return 0.0  # Return 0 if conditions for blink are not met

def gen_frames():
    global COUNTER, TOTAL_BLINKS, start_time, video_capture, video_feed_active

    while True:
        if video_feed_active:
            if video_capture is None or not video_capture.isOpened():
                video_capture = cv2.VideoCapture(0)
                start_time = time.time()
                COUNTER = 0
                TOTAL_BLINKS = 0

            success, frame = video_capture.read()
            if not success:
                break

            frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coordinates = landmarksDetection(frame, results, True)
                eyes_ratio = blinkRatio(frame, mesh_coordinates, RIGHT_EYE, LEFT_EYE)
                cv2.putText(frame, "Please blink your eyes", (int(frame_width / 2), 100), FONT, 1, (0, 255, 0), 2)

                if eyes_ratio > 3.0:  # Adjusted threshold for blink detection
                    COUNTER += 1
                else:
                    if COUNTER > 5:  # Adjusted threshold for counting a blink
                        TOTAL_BLINKS += 1
                        COUNTER = 0

                cv2.rectangle(frame, (20, 120), (290, 160), (0, 0, 0), -1)
                cv2.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (30, 150), FONT, 1, (0, 255, 0), 2)

            elapsed_time = time.time() - start_time
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            cv2.putText(frame, elapsed_time_str, (10, 30), FONT, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        else:
            time.sleep(1)  # If not active, wait for 1 second and check again

    if video_capture:
        video_capture.release()
        cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global video_feed_active
    if not video_feed_active:
        video_feed_active = True
    return "Camera Started"

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global video_feed_active
    video_feed_active = False
    return "Camera Stopped"

if __name__ == '__main__':
    app.run(debug=True)

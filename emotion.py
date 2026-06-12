import os
import cv2
import csv
import urllib.request
import time
from datetime import datetime
from collections import Counter

# 🔥 OPTIMIZATION 1: Suppress all backend TensorFlow/Keras warning logs completely
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def open_working_camera(indices=(0, 1, 2), backends=None):
    if backends is None:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for idx in indices:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            got_frame = False
            for _ in range(8):
                success, frame = cap.read()
                if success and frame is not None and frame.size > 0:
                    got_frame = True
                    break

            if got_frame:
                return cap, idx, backend

            cap.release()

    return None, None, None

script_dir = os.path.dirname(os.path.abspath(__file__))

def init_face_detector(base_dir):
    try:
        import mediapipe as mp

        model_dir = os.path.join(base_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "blaze_face_short_range.tflite")

        if not os.path.exists(model_path):
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/face_detector/"
                "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
            )
            print("Downloading MediaPipe face model...")
            urllib.request.urlretrieve(model_url, model_path)

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=0.5,
        )
        detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        print("✅ Using MediaPipe Tasks face detector")
        return "mediapipe", detector, mp
    except Exception as e:
        print(f"⚠️ MediaPipe unavailable ({e}). Falling back to OpenCV Haar.")
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(face_cascade_path)
        if cascade.empty():
            raise RuntimeError(f"Could not load face cascade from: {face_cascade_path}")
        print("✅ Using OpenCV Haar face detector")
        return "opencv", cascade, None

face_backend, face_detection, mp_module = init_face_detector(script_dir)

cap, selected_index, selected_backend = open_working_camera()
if cap is None:
    raise RuntimeError("Could not open a working webcam feed. Close other camera apps and try again.")

# Set standard venue resolution (Balances crystal clear tracking with performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

csv_file = os.path.join(script_dir, "event_emotion_analytics.csv")
emotion_file = os.path.join(script_dir, "emotion_records.csv")

csv_ready = False
for attempt in range(3):
    try:
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Timestamp", "Detected_Emotion", "Confidence"])
        with open(emotion_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Timestamp", "Emotion"])
        csv_ready = True
        break
    except PermissionError:
        if attempt < 2:
            print(f"⚠️ CSV file locked, retrying...")
            time.sleep(1)
        else:
            print("⚠️ Continuing without logging")
            csv_file = None
            emotion_file = None

print(f"🚀 Optimized 4-Hour Venue Tracker Active! Saving to: {csv_file}")

window_name = 'Live Venue Emotion Analytics Pipeline'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600) # 🔥 OPTIMIZATION 2: Leaner display footprint

try:
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
except Exception:
    pass

frame_count = 0
current_emotion = "Scanning..."
current_confidence = 0
last_status = "Waiting for face..."
emotion_counts = Counter()
deepface_analyze = None
deepface_ready = False
consecutive_read_failures = 0
max_read_failures_before_reconnect = 20
analysis_start_frame = 30

last_analysis_time = 0 

while True:
    if cap is None or not cap.isOpened():
        cap, selected_index, selected_backend = open_working_camera()
        if cap is None:
            break
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        consecutive_read_failures = 0

    success, frame = cap.read()
    if not success or frame is None or frame.size == 0:
        consecutive_read_failures += 1
        if consecutive_read_failures >= max_read_failures_before_reconnect:
            cap.release()
            cap = None
        continue

    consecutive_read_failures = 0
    frame_count += 1
    frame = cv2.flip(frame, 1) 
    h, w, _ = frame.shape

    faces = []
    if face_backend == "mediapipe":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb_frame)
        mp_result = face_detection.detect(mp_image)
        for detection in mp_result.detections:
            bbox = detection.bounding_box
            faces.append((int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)))
    else:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_detection.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for item in detected:
            faces.append(item)

    current_time = time.time()

    if len(faces) > 0:
        for (xmin, ymin, width, height) in faces:
            xmax, ymax = min(w, xmin + width), min(h, ymin + height)
            xmin, ymin = max(0, xmin), max(0, ymin)

            face_roi = frame[ymin:ymax, xmin:xmax]

            if frame_count < analysis_start_frame:
                last_status = "Warming up..."
            
            # Strict 1-second pulse logic
            elif current_time - last_analysis_time >= 1.0 and face_roi.shape[0] > 32 and face_roi.shape[1] > 32:
                try:
                    last_status = "Analyzing..."

                    if not deepface_ready:
                        from deepface import DeepFace
                        deepface_analyze = DeepFace.analyze
                        deepface_ready = True

                    rgb_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    rgb_face_roi = cv2.resize(rgb_face_roi, (224, 224))
                    
                    # 🔥 OPTIMIZATION 3: Force DeepFace to silent execution mode (enforce_detection=False)
                    analysis = deepface_analyze(rgb_face_roi, actions=['emotion'], enforce_detection=False)

                    if isinstance(analysis, list):
                        analysis = analysis[0]

                    current_emotion = analysis['dominant_emotion']
                    current_confidence = round(analysis['emotion'][current_emotion], 2)
                    last_status = f"Active Tracking"
                    emotion_counts[current_emotion] += 1
                    print(f"⏱️ [LOGGED] {datetime.now().strftime('%H:%M:%S')} -> {current_emotion} ({current_confidence}%)")

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if csv_file:
                        try:
                            with open(csv_file, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([timestamp, current_emotion, current_confidence])
                        except Exception:
                            pass

                    if emotion_file:
                        try:
                            with open(emotion_file, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([timestamp, current_emotion])
                        except Exception:
                            pass

                    last_analysis_time = current_time

                except Exception as e:
                    pass

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text_y = ymin - 10 if ymin > 20 else ymin + 25
            cv2.putText(frame, f"{current_emotion}", (xmin, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Render dashboard text labels cleanly
    cv2.putText(frame, f"System Status: {last_status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Analytics Feed: {current_emotion} ({current_confidence}%)", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    y_offset = 100
    for emotion_name, count in emotion_counts.most_common(4):
        cv2.putText(frame, f"{emotion_name}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        y_offset += 25

    cv2.imshow(window_name, frame)

    # 🔥 OPTIMIZATION 4: Micro-sleep parameter to completely stop laptop cpu spiking
    time.sleep(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if face_backend == "mediapipe":
    try: face_detection.close()
    except Exception: pass

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
print(f"✅ Session safely logged to {csv_file}")
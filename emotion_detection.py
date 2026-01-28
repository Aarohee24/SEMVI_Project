
import cv2
from deepface import DeepFace

print("Starting Emotion Detection...")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Camera opened")

frame_count = 0
emotion = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if frame_count % 20 == 0:
            face = frame[y:y+h, x:x+w]
            try:
                result = DeepFace.analyze(
                    face,
                    actions=['emotion'],
                    enforce_detection=False
                )
                emotion = result[0]['dominant_emotion']
                print("Emotion:", emotion)
            except Exception as e:
                print("DeepFace error:", e)

        if emotion:
            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    cv2.imshow("Emotion Detection", frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing...")
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed")

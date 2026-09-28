import streamlit as st
import cv2
import tempfile
import os
from collections import defaultdict
from ultralytics import YOLO

st.set_page_config(page_title="Real-Time Vehicle Counter", page_icon="🚗", layout="wide")
st.title("🚗 Real-Time Vehicle Counter")
st.write("Detect and count vehicles crossing a defined line using YOLO11n + Object Tracking.")

MODEL_PATH = "yolo11n.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("yolo11n.pt was not found. Place it in the same folder as app.py.")
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_file.write(uploaded_file.getbuffer())
    input_file.close()

    st.subheader("Input Video")
    st.video(input_file.name)

    if st.button("🚦 Process Video", type="primary"):
        cap = cv2.VideoCapture(input_file.name)
        if not cap.isOpened():
            st.error("Unable to open the uploaded video.")
            os.unlink(input_file.name)
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        line_y = height // 2

        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = output_file.name
        output_file.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            os.unlink(input_file.name)
            os.unlink(output_path)
            st.error("Unable to create the output video.")
            st.stop()

        previous_positions = {}
        counted_ids = set()
        vehicle_counts = defaultdict(int)
        progress = st.progress(0)
        status = st.empty()
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            results = model.track(frame, persist=True, conf=0.35, verbose=False)
            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                coordinates = result.boxes.xyxy.cpu().numpy()

                for box, track_id, class_id in zip(coordinates, track_ids, class_ids):
                    if class_id not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    vehicle_name = VEHICLE_CLASSES[class_id]

                    if track_id in previous_positions:
                        previous_y = previous_positions[track_id]
                        crossed = previous_y < line_y and center_y >= line_y
                        if crossed and track_id not in counted_ids:
                            vehicle_counts[vehicle_name] += 1
                            counted_ids.add(track_id)

                    previous_positions[track_id] = center_y

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{vehicle_name} ID:{track_id}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

            cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 3)
            cv2.putText(frame, "COUNTING LINE", (20, max(line_y - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            y_position = 40
            for vehicle_name in VEHICLE_CLASSES.values():
                cv2.putText(frame, f"{vehicle_name}: {vehicle_counts[vehicle_name]}",
                            (20, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 255), 2)
                y_position += 32

            out.write(frame)

            if total_frames > 0:
                progress.progress(min(frame_number / total_frames, 1.0))
            status.write(f"Processing frame {frame_number} / {total_frames}")

        cap.release()
        out.release()
        progress.progress(1.0)
        status.success("Video processing completed!")

        st.subheader("Vehicle Count Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cars", vehicle_counts["car"])
        c2.metric("Motorcycles", vehicle_counts["motorcycle"])
        c3.metric("Buses", vehicle_counts["bus"])
        c4.metric("Trucks", vehicle_counts["truck"])
        st.metric("Total Vehicles", sum(vehicle_counts.values()))

        st.subheader("Processed Video")
        st.video(output_path)

        with open(output_path, "rb") as video_file:
            st.download_button("⬇️ Download Processed Video",
                               data=video_file,
                               file_name="vehicle_count_output.mp4",
                               mime="video/mp4")

        try:
            os.unlink(input_file.name)
        except OSError:
            pass

import cv2
import numpy as np
import streamlit as st
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Function to process frames and annotate
def process_frame(frame, model, vehicles):
    count = 0
    total_boxes = 0
    results = model.predict(frame, verbose=False)
    annotator = Annotator(frame)

    for result in results:
        for box in result.boxes:
            total_boxes += 1
            c = box.cls
            if int(c) in vehicles:
                b = box.xyxy[0]
                annotator.box_label(b, model.names[int(c)])
                count += 1

    frame = annotator.result()
    return frame, count, total_boxes

def main():
    st.title("Parking spot detection using YOLO")

    ip_address = st.text_input("Enter the IP address URL of the camera:")
    total_spaces = st.number_input("Enter the total number of parking spaces:", min_value=1, value=10)

    if not ip_address:
        st.warning("Please enter the IP address URL of the camera.")
        return

    if total_spaces < 1:
        st.warning("Total number of parking spaces must be at least 1.")
        return

    # Load YOLO model
    model = YOLO("yolov8s.pt")
    vehicles = [2, 3, 5, 7]

    cap = cv2.VideoCapture(ip_address)

    # Placeholder for video feed
    video_placeholder = st.empty()

    # Variables for time and frame display
    last_display_time = datetime.now()
    display_interval = 15  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error reading from camera.")
            break

        # Process frame and get count
        frame, count, total_boxes = process_frame(frame, model, vehicles)

        # Calculate available parking spaces
        if count > total_spaces:
            st.error("Detected cars exceed the total available parking spaces. Please correct the input.")
            break
        else:
            available_spaces = total_spaces - count

        # Display video feed
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Display info every 15 seconds
        current_time = datetime.now()
        time_diff = (current_time - last_display_time).total_seconds()
        if time_diff >= display_interval:
            accuracy = 100 * count / total_boxes if total_boxes > 0 else 0
            time_str = current_time.strftime('%H:%M:%S')
            accuracy_str = f"{accuracy:.0f}%" if total_boxes > 0 else "-"
            cols = st.columns([3, 1, 1, 1])
            with cols[0]:
                st.write("Time")
                st.write(time_str)
            with cols[1]:
                st.write("Vehicles")
                st.write(count)
            with cols[2]:
                st.write("Available Parking")
                st.write(available_spaces)
            with cols[3]:
                st.write("Accuracy")
                st.write(accuracy_str)
            last_display_time = current_time

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    main()

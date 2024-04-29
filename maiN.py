import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def main():
    st.title("Parking spot detection using YOLO")
    
    ip_address = st.text_input("Enter the IP address of the camera:")
    
    if not ip_address:
        st.warning("Please enter the IP address of the camera.")
        return
    
    # Variables
    model = YOLO("yolov8s.pt")
    vehicles = [2, 3, 5, 7]

    cap = cv2.VideoCapture(ip_address)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error reading from camera.")
            break

        count = 0
        results = model.predict(frame, verbose=False)
        
        for result in results:
            annotator = Annotator(frame)

            for box in result.boxes:
                c = box.cls
                if int(c) in vehicles:
                    b = box.xyxy[0]
                    annotator.box_label(b, model.names[int(c)])
                    count += 1

        frame = annotator.result()
        
        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        cv2.putText(frame, 'Total Vehicles: {}'.format(str(count)), (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        st.image(frame, channels="BGR", use_column_width=True)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    main()

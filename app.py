import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile

def predict(image_input):
    model=YOLO('runs/detect/train5/weights/best.pt')
    
    # Check if the input is a file path (str) or a numpy array
    if isinstance(image_input, str):
        # If it's a file path, read the image
        image = cv2.imread(image_input)
    else:
        # If it's a numpy array, it's already an image
        image = image_input

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
    
    predictions=model.predict(image)
    prediction=predictions[0]

    coordinates = None  # Initialize coordinates
    for box in prediction.boxes:
        pred=prediction.names[box.cls[0].item()]
        coordinates=box.xyxy[0].tolist()
        confidence=box.conf[0].item()

    if coordinates is not None:  # Check if coordinates were assigned a value  
        x,y,w,h=coordinates[0],coordinates[1],coordinates[2],coordinates[3]
        if pred=='perfect':
            predicted_image= cv2.rectangle(image,(int(x),int(y)),(int(x+w),int(y+h)),(9, 121, 105),2)
            cv2.putText(img=predicted_image, text=pred, org=(int(x), int(y)-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5,color= (9, 121, 105),thickness= 3)
        else:
            predicted_image= cv2.rectangle(image,(int(x),int(y)),(int(x+w),int(y+h)),(215, 0, 64),2)
            cv2.putText(img=predicted_image, text=pred, org=(int(x), int(y)-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5,color= (215, 0, 64),thickness= 3)
        return predicted_image
    else:
        return image  # Return the original image if no boxes were found
    
def main_loop():
    st.title('Coin Defect Detection')
    input_source= st.selectbox('Select input source', ['Webcam', 'Upload a video', 'Upload an image'])
    if input_source=='Upload an image':
        image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
        if not image_file:
            return None

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_file.seek(0)
            image_path = tmp_file.name

        original_image = Image.open(image_path)
        predicted_image = predict(image_path)

        st.text("Original Image vs Detected Image")
        st.image([original_image, predicted_image])

        # Don't forget to delete the temporary file after usage
        os.remove(image_path)
    elif input_source == 'Webcam':
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

        # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict on frame
            predicted_frame = predict(frame)

        # Display original frame and predicted frame
            st.text("Original Frame vs Detected Frame")
            st.image([frame, predicted_frame])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        file_upload = st.file_uploader("Upload a video", type=['mp4'])
        if not file_upload:
            return None
        cap = cv2.VideoCapture(file_upload)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

        # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict on frame
            predicted_frame = predict(frame)

        # Display original frame and predicted frame
            st.text("Original Frame vs Detected Frame")
            st.image([frame, predicted_frame])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main_loop()

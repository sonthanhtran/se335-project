import streamlit as st
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import torchvision
from torch import nn
from mtcnn import MTCNN
import cv2
import numpy as np

# Load your PyTorch model
model = torchvision.models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 7)
weight = torch.load("./best_aug.pth")
model.load_state_dict(weight['model_state_dict'])
model.eval()

# Load MTCNN for face detector
face_detector = MTCNN()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_expression(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

def detect_face(image):
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    
    faces = face_detector.detect_faces(image_np)
    if faces:
        # Extract the first detected face
        x, y, width, height = faces[0]['box']
        face_image = Image.fromarray(image_np[y:y+height, x:x+width])
        return face_image
    else:
        return None

def main():
    st.title("Facial Expression Recognition Demo")
    st.write(
        "Upload an image, and the model will predict the facial expression."
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        original_image = Image.open(uploaded_file)

        # Detect face
        face_image = detect_face(original_image)

        if face_image is not None:
            # Get bounding box coordinates
            faces = face_detector.detect_faces(np.array(original_image))
            x, y, width, height = faces[0]['box']

            # Draw bounding box on the original image
            annotated_image = original_image.copy()
            draw = ImageDraw.Draw(annotated_image)
            draw.rectangle([x, y, x + width, y + height], outline="red", width=2)

            # Use Streamlit columns to display images side by side
            col1, col2 = st.columns(2)

            # Display the original image in the first column
            col1.image(original_image, caption="Uploaded Image", use_column_width=True)

            # Display the annotated image in the second column
            col2.image(annotated_image, caption="Detected Face with Bounding Box", use_column_width=True)

            # Make prediction
            prediction = predict_expression(face_image)
            expression_mapping = {
                0: "Angry",
                1: "Disgust",
                2: "Fear",
                3: "Happy",
                4: "Sad",
                5: "Surprise",
                6: "Neutral",
            }
            predicted_expression = expression_mapping[prediction]

            st.write(f"Predicted Expression: {predicted_expression}")
        else:
            st.write("No face detected in the uploaded image.")

if __name__ == '__main__':
    main()

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = cap.read()

    fake_frames = 0
    real_frames = 0
    percentage=0
    while success:
        # Convert the frame to PIL Image
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(input_image)

        # Detect face
        face = mtcnn(input_image)
        if face is not None:
            # Preprocess face
            face = face.unsqueeze(0)  # add the batch dimension
            face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
            face = face.to(DEVICE).to(torch.float32) / 255.0

            # Get prediction
            with torch.no_grad():
                output = torch.sigmoid(model(face).squeeze(0))
                prediction = "real" if output.item() < 0.5 else "fake"
                # print(f"Prediction: {prediction} ({output.item():.2f})")
                percentage=percentage+output.item()
                if prediction == "real":
                    real_frames += 1
                else:
                    fake_frames += 1

        success, image = cap.read()

    cap.release()

    total_frames = fake_frames + real_frames
    fake_percentage = (fake_frames / total_frames) * 100
    real_percentage = (real_frames / total_frames) * 100
    percentage=percentage/total_frames*100
    
    print(f"Total frames: {total_frames}")
    print(f"Fake frames: {fake_frames} ({fake_percentage:.2f}%)")
    print(f"Real frames: {real_frames} ({real_percentage:.2f}%)")
    print(f"Fake Average percentage: {percentage:.2f}%")
# Example usage
predict_video("video/fake1.mp4")

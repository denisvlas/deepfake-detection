from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import os
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import cv2
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from flask_cors import CORS
import shutil
from pytube import YouTube


app = Flask(__name__)
CORS(app)

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


def extract_frames_from_youtube(youtube_link, frame_skip=5):
    yt = YouTube(youtube_link)
    stream = yt.streams.filter(file_extension="mp4").first()
    video = cv2.VideoCapture(stream.url)
    
    # Creați directorul "screenshots" dacă nu există deja
    screenshots_dir = "screenshots"
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)
    
    frames = []
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            # Salvați fiecare cadru în directorul "screenshots" cu un nume de fișier unic
            frame_filename = os.path.join(screenshots_dir, f"frame_{frame_count}.png")
            cv2.imwrite(frame_filename, frame)
    video.release()
    return frames

    
def delete_directory_contents(directory):
    try:
        # Iterăm prin toate fișierele și directoarele din director
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                # Dacă este un fișier, îl ștergem
                os.remove(item_path)
            elif os.path.isdir(item_path):
                # Dacă este un director, îl ștergem recursiv
                shutil.rmtree(item_path)
        print(f"Contents of directory '{directory}' have been deleted successfully.")
    except Exception as e:
        print(f"Failed to delete contents of directory '{directory}'. Reason: {e}")


def predict_video(youtube_link):
    # Extrage cadrele din videoclipul YouTube
    frames = extract_frames_from_youtube(youtube_link)

    # Inițializați contoarele pentru cadrele reale și false
    fake_frames = 0
    real_frames = 0
    percentage = 0

    # Parcurgeți fiecare cadru și aplicați logica de detectare și analiză a feței
    for frame in frames:
        # Convertiți cadru la imagine PIL
        input_image = Image.fromarray(frame)

        # Detectați fața
        face = mtcnn(input_image)
        if face is not None:
            # Preprocesați fața
            face = face.unsqueeze(0)
            face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
            face = face.to(DEVICE).to(torch.float32) / 255.0

            # Obțineți predicția
            with torch.no_grad():
                output = torch.sigmoid(model(face).squeeze(0))
                prediction = "real" if output.item() < 0.5 else "fake"
                percentage += output.item()

                # Actualizați contoarele pentru cadrele reale și false
                if prediction == "real":
                    real_frames += 1
                else:
                    fake_frames += 1

    # Calculați procentajele și procentajul mediu de "falsitate"
    total_frames = fake_frames + real_frames
    fake_percentage = (fake_frames / total_frames) * 100
    real_percentage = (real_frames / total_frames) * 100
    percentage = percentage / total_frames * 100

    if real_frames > fake_frames:
        final_prediction = "real"
    else:
        final_prediction = "fake"

    # Afișați rezultatele
    print(f"Total frames: {total_frames}")
    print(f"Fake frames: {fake_frames} ({fake_percentage:.2f}%)")
    print(f"Real frames: {real_frames} ({real_percentage:.2f}%)")
    print(f"Fake Average percentage: {percentage:.2f}%")

    return real_percentage, percentage, final_prediction



@app.route('/predict', methods=['POST'])
def predict():

    data = request.json
    url = data['video_url']
    if not url:
        return jsonify({'error': 'No video URL provided'})
    # print("dsadas",request.form)

    
        # Predict frames
    real_percentage, percentage, final_prediction = predict_video(url)
    delete_directory_contents("screenshots")
    return jsonify({
        'success': 'Screenshots taken successfully',
        'percentage': percentage,
        'real_percentage': real_percentage,
        'final_prediction': final_prediction
    })

if __name__ == '__main__':
    app.run(debug=True)

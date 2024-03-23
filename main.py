from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image


from flask import Flask, request, jsonify
from selenium import webdriver
import time
import os
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from pytube import YouTube
from io import BytesIO
import cv2
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from flask import Flask, request, jsonify
from flask_cors import CORS

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
def take_screenshots(video_url):
    capture_interval = 0.2

    # Configure Chrome driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Open YouTube video
    driver.get(video_url)
    time.sleep(2)  # Wait for the video to start
    # Play the video
    driver.find_element(By.CSS_SELECTOR, ".ytp-play-button").click()

    # Loop for image capture
    timestamp = int(time.time())  # Use a unique timestamp for filenames
    frame_count = 0
    video_div = driver.find_element(By.CSS_SELECTOR, "div.html5-video-player")
    time.sleep(2)  # Wait for the video to start

    for i in range(20):
        # Wait for the capture interval
        time.sleep(capture_interval)

        # Get a screenshot of the browser window
        screenshot = video_div.screenshot_as_png

        # Convert the screenshot to a NumPy array
        image_array = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        screenshotDir="screenshots"
        # Save the image with a unique filename
        filename = f"{screenshotDir}/frame_{timestamp}_{i}.png"
        cv2.imwrite(filename, image_array)

        frame_count += 1

    # Close the browser
    driver.quit()

    return screenshotDir


    

import os

def predict_frames(frames_dir):
    fake_frames = 0
    real_frames = 0
    percentage = 0
    total_frames = 0

    for filename in os.listdir(frames_dir):
        if filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(frames_dir, filename)
            image = cv2.imread(image_path)

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
                    percentage += output.item()
                    if prediction == "real":
                        real_frames += 1
                    else:
                        fake_frames += 1

            total_frames += 1

    # Calculate percentages
    fake_percentage = (fake_frames / total_frames) * 100
    real_percentage = (real_frames / total_frames) * 100
    percentage = percentage / total_frames * 100

    print(f"Total frames: {total_frames}")
    print(f"Fake frames: {fake_frames} ({fake_percentage:.2f}%)")
    print(f"Real frames: {real_frames} ({real_percentage:.2f}%)")
    print(f"Fake Average percentage: {percentage:.2f}%")

    return fake_percentage, real_percentage, percentage



@app.route('/predict', methods=['POST'])
def predict():

    data = request.json
    url = data['video_url']
    if not url:
        return jsonify({'error': 'No video URL provided'})
    # print("dsadas",request.form)

    screenshots_dir = take_screenshots(url)

        # Predict frames
    fake_percentage, real_percentage, percentage = predict_frames(screenshots_dir)

    return jsonify({
        'success': 'Screenshots taken successfully',
        'fake_percentage': fake_percentage,
        'real_percentage': real_percentage,
        'average_percentage': percentage
    })

if __name__ == '__main__':
    app.run(debug=True)

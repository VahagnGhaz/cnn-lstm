import torch
import os
import cv2 as cv
import numpy as np
from model import CNNLSTM
from utils import load_params
from albumentations.pytorch import ToTensorV2
import albumentations as A

activity_labels = {
    0: "Biking",
    1: "CliffDiving",
    2: "Drumming",
    3: "Haircut",
    4: "HandstandWalking",
    5: "HighJump",
    6: "JumpingJack",
    7: "Mixing",
    8: "Skiing",
    9: "SumoWrestling"
}

def read_video(video_path, num_frames, transform, img_size):
    frames = []
    cap = cv.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, img_size)  # Resize the frame
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert to RGB
        if transform:
            transformed = transform(image=frame)
            frame = transformed['image']
        frames.append(frame)

    cap.release()

    # Handling case where video has fewer frames than num_frames
    while len(frames) < num_frames:
        frames.extend(frames[:num_frames - len(frames)])

    video_tensor = torch.stack(frames, dim=0)  # Shape: num_frames x H x W x C
    video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension: 1 x num_frames x C x H x W

    return video_tensor


def infer_single_video(model, video_tensor, loss_fn, device):
    model.to(device)
    video_tensor = video_tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(video_tensor)
        _, predicted = torch.max(outputs.data, 1)
        loss = loss_fn(outputs, predicted.long()).item()
    return predicted.item(), loss

# Load parameters and model
params = load_params("params.json")
device = params['device']
model = CNNLSTM(num_classes=params['num_classes'], hidden_size=params['hidden_size'], num_lstm_layers=params['num_lstm_layers'], use_pretrained=params['use_pretrained']).to(device)
model_path = os.path.join(params['cache_dir'], params['best_weights'])
model.load_state_dict(torch.load(model_path, map_location=device))

# Define the transformation
transform = A.Compose([A.Resize(height=params['img_size'][0], width=params['img_size'][1]), A.Normalize(), ToTensorV2()])

# Load the video
video_path = 'resources/demo.mp4'
video_tensor = read_video(video_path, params['num_frames'], transform, params['img_size'])

# Perform inference
loss_fn = torch.nn.CrossEntropyLoss()
predicted_label, loss = infer_single_video(model, video_tensor, loss_fn, device)
print(f"Predicted Label: {activity_labels[int(predicted_label)]}")


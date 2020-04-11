#!/usr/bin/env python

import cv2
import numpy as np
import torch
from PIL import Image

model = torch.hub.load('pytorch/vision:v0.4.2', 'mobilenet_v2', pretrained=True)
model.eval()

from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load Imagenet Synsets
with open('imagenet_synsets.txt', 'r') as f:
    synsets = f.readlines()
synsets = [x.strip() for x in synsets]
splits = [line.split(' ') for line in synsets]
key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}

with open('imagenet_classes.txt', 'r') as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

cap=cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        pil_frame = Image.fromarray(frame);
        input_tensor = preprocess(pil_frame)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        with torch.no_grad():
            output = model(input_batch)
            confidence, argmax = output[0].max(0)
            class_id = argmax
            class_key = class_id_to_key[class_id]
            classname = key_to_classname[class_key]
            cv2.putText(frame, "'{}' : '{}'".format(classname, confidence), (0,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27, 32, ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()

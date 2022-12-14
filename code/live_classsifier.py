"""
Application that uses the CNN model to classify ASL motions
in real-time using webcam video.

Usage: > python live_classifier.py

"""
import cv2
import torch
from torchvision import transforms
import PIL

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])
model = torch.load("googlenet_asl_v2.pth",
                   map_location=torch.device('cpu'))
videoCapture = cv2.VideoCapture(0)
classes = ['A',
           'B',
           'C',
           'D',
           'E',
           'F',
           'G',
           'H',
           'I',
           'J',
           'K',
           'L',
           'M',
           'N',
           'O',
           'P',
           'Q',
           'R',
           'S',
           'T',
           'U',
           'V',
           'W',
           'X',
           'Y',
           'Z',
           'del',
           'nothing',
           'space']

i = 0
while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    # resize height and width to 200
    resized = cv2.resize(frame, (200, 200))
    if i % 10 == 0:  # classify frames at small intervals
        image = PIL.Image.fromarray(resized)  # get the image from array
        input_tensor = preprocess(image).float()  # process the image
        # unsqueeze since model works with batches
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(input_batch)  # get the output
            prediction = output.argmax(dim=1)  # get the prediction
            print(classes[prediction])  # get the class from the prediction
    cv2.imshow('frame', resized)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()

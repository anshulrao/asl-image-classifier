import cv2
import torch
from torchvision import transforms
import PIL

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])
model = torch.load("googlenet_asl_v2.pth", map_location=torch.device('cpu'))
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

    resized = cv2.resize(frame, (200, 200))
    if i % 10 == 0:
        image = transform(PIL.Image.fromarray(resized)).float()
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            prediction = output.argmax(dim=1)
            print(classes[prediction])

    cv2.imshow('frame', resized)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()

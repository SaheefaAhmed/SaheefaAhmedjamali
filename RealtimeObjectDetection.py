import cv2 as cv
import torch
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# index streaming
idx_streaming = 0
flip=True
flip_code=1

# initialize streaming capture
print('acquisition initialization..')
videoCapture = cv.VideoCapture(idx_streaming, cv.CAP_DSHOW)
videoCapture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
videoCapture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
print('acquisition started!')
print('(press "q" if you want to exit or force stop)')

# check if streaming is active
if not videoCapture.isOpened():
    print('streaming is no more available')
    exit()

# capture images
while True:
    # grab frame from streaming
    acquisition_done, frame = videoCapture.read()

    # check if frame has been acquired correctly
    if not acquisition_done:
        print('unable to acquire a frame from streaming')
        break

    # flip if needed
    if flip:
        frame = cv.flip(src=frame, flipCode=flip_code)

    # yolo detection
    results = model(Image.fromarray(frame))
    results = results.pandas().xyxy[0]

    for idx, result in results.iterrows():    
        cv.rectangle(frame, (int(result['xmin']), int(result['ymin'])), (int(result['xmax']), int(result['ymax'])), (0, 255, 0), 2)
        cv.putText(frame, f"{result['name']} {int(result['confidence']*100)}%", (int(result['xmin']), int(result['ymin']) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # show streaming        
    cv.imshow('frame', frame)

    # exit
    if cv.waitKey(1) == ord('q'):
        break

# close streaming
videoCapture.release()
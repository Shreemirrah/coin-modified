from ultralytics import YOLO
from fastapi import FastAPI,File,UploadFile
import numpy as np
from PIL import Image
from io import BytesIO
import uvicorn
import cv2 
from fastapi.responses import JSONResponse



model=YOLO('runs/detect/train5/weights/best.pt')


app=FastAPI()

@app.get('/')

def main():
    print("Starting server...")

@app.get("/{name}")

def printName(name: str):
    return {f"Welcome,{name}"}
'''
def letterbox(img, new_shape=(800,800), color=(128,128,128), auto=True, scaleFill=False, scaleUp=True):
    shape = img.shape[:2] # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleUp: # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # compute padding
    ratio = r, r # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] # wh padding
    if auto: # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64) # wh padding
    elif scaleFill: # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[0], new_shape[1] / shape[1] # width, height ratios

    dw /= 2 # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad: # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    if top < 0 or bottom < 0 or left < 0 or right < 0:
        raise ValueError(f"The calculated values for top, bottom, left, and right are negative. Top: {top}, Bottom: {bottom}, Left: {left}, Right: {right}")

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) # add border
    return img, ratio, (dw, dh)

def read_image(data,size=(1920,1080))->np.ndarray:
    img=Image.open(BytesIO(data))
    img, _, (dw, dh) = letterbox(np.array(img), new_shape=size)
    return img





def read_image(data,size=(1920,1080))->np.ndarray:
    img=Image.open(BytesIO(data))
    img=img.resize(size,Image.Resampling.BILINEAR)
    return np.array(img)

'''
def read_image(data)->np.ndarray:
    img=Image.open(BytesIO(data))
    img=img.resize((1920,1080),Image.Resampling.BILINEAR)
    return np.array(img)



@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    contents=await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    predictions=model.predict(img)
    prediction=predictions[0]
    for box in prediction.boxes:
        pred=prediction.names[box.cls[0].item()]
        coordinates=box.xyxy[0].tolist()
        confidence=box.conf[0].item()

    result = {
        "bounding_boxes": coordinates,
        "classes": pred,
        "scores": confidence,
    }
    return JSONResponse(content=result)
'''
    predictions=model.predict(contents)
    image=read_image(contents)
    result = {
        "bounding_boxes": predictions.xyxy[0].tolist(),
        "classes": predictions.names[0].tolist(),
        "scores": predictions.xyxy[0][:, 4].tolist(),
    }
'''
    
   # image=np.expand_dims(image,0)
    #  img=image.tolist()
    
   # return prediction
   # return image.tolist()
   # return cv2img

if __name__=='__main__':
    uvicorn.run(app, host="localhost", port=8000)
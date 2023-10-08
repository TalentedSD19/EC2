from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
import numpy as np
import base64
from fastapi.middleware.cors import CORSMiddleware
import keys
from twilio.rest import Client
import requests

app = FastAPI()
client = Client(keys.account_sid, keys.auth_token)
# Allow requests from any origin
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageData(BaseModel):
    imagedata: str


class Location(BaseModel):
    latitude: str
    longitude: str
    address: str


class Coordinates(BaseModel):
    latitude: str
    longitude: str


# Load YOLOv5 model
weight1 = 'curr2.pt'
weight2 = 'face2.pt'
weight3 = 'yolov5m6.pt'
device = select_device('')
model1 = attempt_load(weight1, device)
model2 = attempt_load(weight2, device)
model3 = attempt_load(weight3, device)
stride1 = int(model1.stride.max())
stride2 = int(model2.stride.max())
stride3 = int(model3.stride.max())


@app.post("/currency")
def currency(imagedata: ImageData):
    try:
        # Decode base64 string and read it as an image
        image_bytes = base64.b64decode(imagedata.imagedata)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = model1(img)[0]
        pred = non_max_suppression(pred, 0.45, 0.5)

        recognized_currency = []

        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det

            cls = int(cls.item())

            if cls == 0:
                recognized_currency.append('This is a Ten Rupees note')
            elif cls == 1:
                recognized_currency.append('This is a Twenty Rupees note')
            elif cls == 2:
                recognized_currency.append('This is a Fifty Rupees note')
            elif cls == 3:
                recognized_currency.append('This is a Hundred Rupees note')
            elif cls == 4:
                recognized_currency.append('This is a Two Hundred Rupees note')
            elif cls == 5:
                recognized_currency.append(
                    'This is a Five Hundred Rupees note')
            elif cls == 6:
                recognized_currency.append(
                    'This is a Two Thousand Rupees note')

        if not recognized_currency:
            recognized_currency.append(
                'This note cannot be recognized. Please hold it correctly.')

        return recognized_currency
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/currency_bangla")
def currency(imagedata: ImageData):
    try:
        # Decode base64 string and read it as an image
        image_bytes = base64.b64decode(imagedata.imagedata)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = model1(img)[0]
        pred = non_max_suppression(pred, 0.45, 0.5)

        recognized_currency = []

        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det

            cls = int(cls.item())

            if cls == 0:
                recognized_currency.append('এটি দশ টাকার নোট')
            elif cls == 1:
                recognized_currency.append('এটি কুড়ি টাকার নোট')
            elif cls == 2:
                recognized_currency.append('এটি পঞ্চাশ টাকার নোট')
            elif cls == 3:
                recognized_currency.append('এটি একশো টাকার নোট')
            elif cls == 4:
                recognized_currency.append('এটি দুশো টাকার নোট')
            elif cls == 5:
                recognized_currency.append(
                    'এটি পাঁচশো টাকার নোট')
            elif cls == 6:
                recognized_currency.append(
                    'এটি দুই হাজার টাকার নোট')

        if not recognized_currency:
            recognized_currency.append(
                'এই নোট টি চিহ্নিত করা যাচ্ছে না   ঠিক ভাবে তুলে ধরুন')

        return recognized_currency
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/currency_hindi")
def currency(imagedata: ImageData):
    try:
        # Decode base64 string and read it as an image
        image_bytes = base64.b64decode(imagedata.imagedata)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = model1(img)[0]
        pred = non_max_suppression(pred, 0.45, 0.5)

        recognized_currency = []

        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det

            cls = int(cls.item())

            if cls == 0:
                recognized_currency.append('ये दस रुपये का नोट है')
            elif cls == 1:
                recognized_currency.append('यह बीस रुपये का नोट है')
            elif cls == 2:
                recognized_currency.append('यह पचास रुपये का नोट है')
            elif cls == 3:
                recognized_currency.append('ये सौ रुपये का नोट है')
            elif cls == 4:
                recognized_currency.append('यह दो सौ रुपये का नोट है')
            elif cls == 5:
                recognized_currency.append(
                    'यह पांच सौ रुपये का नोट है')
            elif cls == 6:
                recognized_currency.append(
                    'यह दो हजार रुपये का नोट है')

        if not recognized_currency:
            recognized_currency.append(
                'इस नोट को पहचाना नहीं जा सकता. कृपया इसे सही ढंग से पकड़ें')

        return recognized_currency
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/facerecognize")
async def facerecognize(imagedata: ImageData):
    recognized_faces = []
    try:
        # Decode base64 string and read it as an image
        image_bytes = base64.b64decode(imagedata.imagedata)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = model2(img)[0]
        pred = non_max_suppression(pred, 0.45, 0.5)

        recognized_faces = []

        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls.item())
            if cls == 0:
                recognized_faces.append('This is Soumyajit')
            elif cls == 1:
                recognized_faces.append('This is Debaditya')

        if not recognized_faces:
            recognized_faces.append(
                'This person cannot be recognized.')

        return recognized_faces
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/facerecognize_bangla")
async def facerecognize(imagedata: ImageData):
    recognized_faces = []
    try:
        # Decode base64 string and read it as an image
        image_bytes = base64.b64decode(imagedata.imagedata)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = model2(img)[0]
        pred = non_max_suppression(pred, 0.45, 0.5)

        recognized_faces = []

        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls.item())
            if cls == 0:
                recognized_faces.append('ইনি সৌম্যজিৎ')
            elif cls == 1:
                recognized_faces.append('ইনি দেবাদিত্য')

        if not recognized_faces:
            recognized_faces.append(
                'এই ব্যক্তিকে চেনা যাচ্ছে না')

        return recognized_faces
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/facerecognize_hindi")
async def facerecognize(imagedata: ImageData):
    recognized_faces = []
    try:
        # Decode base64 string and read it as an image
        image_bytes = base64.b64decode(imagedata.imagedata)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = model2(img)[0]
        pred = non_max_suppression(pred, 0.45, 0.5)

        recognized_faces = []

        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls.item())
            if cls == 0:
                recognized_faces.append('ये सौम्यजीत हैं')
            elif cls == 1:
                recognized_faces.append('यह देबदित्य है')

        if not recognized_faces:
            recognized_faces.append(
                'इस शख्स की पहचान नहीं हो पा रही है')

        return recognized_faces
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/roadassist')
async def predict(imagedata: ImageData):
    formatted_outputs = []
    desired_classes = {0: 'person',
                       1: 'bicycle',
                       2: 'car',
                       3: 'motorcycle',
                       5: 'bus',
                       7: 'truck',
                       9: 'traffic light',
                       10: 'fire hydrant',
                       11: 'stop sign',
                       15: 'cat',
                       16: 'dog',
                       17: 'horse',
                       19: 'cow'}
    try:
        # Decode base64 string and read it as an image
        image_bytes = base64.b64decode(imagedata.imagedata)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = model3(img)[0]
        pred = non_max_suppression(pred, 0.45, 0.5)

        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls.item())
            if cls in desired_classes.keys():
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                area = (x1 - x2) * (y1 - y2)

                if center_x < 200:
                    side = 'left'
                elif center_x >= 200 and center_x <= 400:
                    side = 'front'
                else:
                    side = 'right'

                name = desired_classes[cls]
                if side == 'front':
                    data = f'{name} in {side} of you'
                else:
                    data = f'{name} to your {side}'
                if data not in formatted_outputs and area > 8000:
                    formatted_outputs.append(data)

        return formatted_outputs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/roadassist_bangla')
async def predict(imagedata: ImageData):
    formatted_outputs = []
    desired_classes = {0: 'মানুষ',
                       1: 'সাইকেল',
                       2: 'গাড়ি',
                       3: 'বাইক',
                       5: 'বাস',
                       7: 'ট্রাক',
                       9: 'ট্র্যাফিক লাইট',
                       10: 'অগ্নি নিধরক',
                       11: 'স্টপ চিহ্ন',
                       15: 'বিড়াল',
                       16: 'কুকুর',
                       17: 'ঘোড়া',
                       19: 'গরু'}
    try:
        # Decode base64 string and read it as an image
        image_bytes = base64.b64decode(imagedata.imagedata)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = model3(img)[0]
        pred = non_max_suppression(pred, 0.45, 0.5)

        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls.item())
            if cls in desired_classes.keys():
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                area = (x1 - x2) * (y1 - y2)

                if center_x < 200:
                    side = 'বাম'
                elif center_x >= 200 and center_x <= 400:
                    side = 'সামনের'
                else:
                    side = 'ডান'

                name = desired_classes[cls]

                data = f'{side} দিকে {name}'
                if data not in formatted_outputs and area > 8000:
                    formatted_outputs.append(data)

        return formatted_outputs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/roadassist_hindi')
async def predict(imagedata: ImageData):
    formatted_outputs = []
    desired_classes = {
        0: 'आदमी',
        1: 'साइकिल',
        2: 'गाड़ी',
        3: 'बाइक',
        5: 'बस',
        7: 'ट्रक',
        9: 'ट्रैफिक लाइट',
        10: 'अग्नि निधरक',
        11: 'स्टॉप साइन',
        15: 'बिल्ली',
        16: 'कुत्ता',
        17: 'घोड़ा',
        19: 'गाय'
    }

    try:
        # Decode base64 string and read it as an image
        image_bytes = base64.b64decode(imagedata.imagedata)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = model3(img)[0]
        pred = non_max_suppression(pred, 0.45, 0.5)

        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls.item())
            if cls in desired_classes.keys():
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                area = (x1 - x2) * (y1 - y2)

                if center_x < 200:
                    side = 'बायीं तरफ'
                elif center_x >= 200 and center_x <= 400:
                    side = 'सामने की ओर'
                else:
                    side = 'दाहिनी तरफ'

                name = desired_classes[cls]

                data = f'{side} {name}'
                if data not in formatted_outputs and area > 8000:
                    formatted_outputs.append(data)

        return formatted_outputs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/smsalert")
async def smsalert(location: Location):
    client.messages.create(
        body=f'''

ALERT: Visually Impaired Distress 🚨


Location: {location.address}

Coordinates: Latitude {location.latitude}, Longitude {location.longitude}


This is an automated alert. A visually impaired person is in distress at the above location. Immediate assistance required.

BlindSight AI''',
        from_=keys.twillio_number,
        to=keys.recepient_number
    )


@app.post("/weather")
def weather(coordinates: Coordinates):
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={coordinates.latitude}&lon={coordinates.longitude}&appid=dbb8a98372740af7be4ed458550fbb2a'

    # Send the GET request
    response = requests.get(url)
    print('got it')
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        return data
    else:
        return 'Error fetching weather data'

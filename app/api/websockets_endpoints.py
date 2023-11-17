import cv2
from ultralytics import YOLO
import math
import numpy
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
from base64 import b64encode
import time

ws_router = APIRouter()

detector = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./Haarcascades/haarcascade_eye.xml')

model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


@ws_router.websocket("/check_ws_connection")
async def check_ws_connection(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_text()
        # Perform any necessary checks or operations on the WebSocket data

        # Example: Send a message back to the WebSocket client
        await websocket.send_text("WebSocket connection is active and checked.")
    except WebSocketDisconnect as e:
        print("WebSocket connection closed unexpectedly")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()


# best performance for sending processed images
@ws_router.websocket("/ws-fast")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    processing_times = []
    frame_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()

            start_time = time.time()  # Start time of processing

            # Convert to a numpy array
            nparr = numpy.frombuffer(data, numpy.uint8)

            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                # Perform image processing as before
                faces = detector.detectMultiScale(frame, 1.3, 5)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 3)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                _, buffer = cv2.imencode('.jpg', frame)
                frame_encoded = b64encode(buffer).decode('utf-8')
                processing_time = time.time() - start_time  # End time of processing
                processing_times.append(processing_time)
                frame_counter += 1
                print(f"Processing time: {processing_time} seconds")

                # aici le trimite inapoi
                await websocket.send_text('data:image/jpeg;base64,' + frame_encoded)
            else:
                print('Frame is None')
    except WebSocketDisconnect:
        get_time(processing_times, frame_counter)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()


# Function used for getting the average time for a frame render
def get_time(processing_times=None, frame_counter=None):
    total_processing_time = sum(processing_times)
    print(f"Total processing time for all images: {total_processing_time} seconds")
    print(f"Total frames processed: {frame_counter}")

    if frame_counter > 0:
        average_processing_time = total_processing_time / frame_counter
        print(f"Average processing time per frame: {average_processing_time} seconds")
    else:
        print("No frames were processed.")


@ws_router.websocket("/ws-yolo")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    processing_times = []
    frame_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = numpy.frombuffer(data, numpy.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            results = model(img)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    # class name
                    cls = int(box.cls[0])
                    print("Class name -->", classNames[cls])

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(img, classNames[cls], org, font, font_scale, color, thickness)

                # Convert the image with boxes back to bytes
            _, buffer = cv2.imencode('.jpg', img)
            frame_encoded = b64encode(buffer).decode('utf-8')

            data_to_send = {
                "image": 'data:image/jpeg;base64,' + frame_encoded
            }

            await websocket.send_json(data_to_send)
    except WebSocketDisconnect:
        await websocket.close()
        get_time(processing_times, frame_counter)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()



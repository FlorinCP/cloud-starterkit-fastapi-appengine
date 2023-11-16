import cv2
import numpy
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
from base64 import b64encode
import time


ws_router = APIRouter()

detector = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./Haarcascades/haarcascade_eye.xml')


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


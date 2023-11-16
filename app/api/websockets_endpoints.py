from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

ws_router = APIRouter()


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

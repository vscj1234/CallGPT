from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import openai
import time
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

OPENAI_API_KEY = "sk-****" # Replace with actual key
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class ConnectionManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_message(self, websocket: WebSocket, message: str, message_type: str = "text"):
        try:
            await websocket.send_json({
                "type": message_type,
                "content": message,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise

manager = ConnectionManager()

async def process_openai_response(websocket: WebSocket, message: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Bluu, an assistant from ScalebuildAI. You help people find the best products. Scalebuild is a software company."},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            stream=True,
            max_tokens=50
        )

        # Initialize variables for collecting response
        current_sentence = []
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                current_sentence.append(content)
                
                # Send complete sentences to maintain natural speech flow
                if any(punct in content for punct in ['.', '!', '?', '\n']):
                    complete_sentence = ''.join(current_sentence).strip()
                    if complete_sentence:
                        await websocket.send_json({
                            "type": "text",
                            "content": complete_sentence,
                            "timestamp": time.time()  # Added timestamp for better tracking
                        })
                        current_sentence = []
        
        # Send any remaining content
        if current_sentence:
            final_content = ''.join(current_sentence).strip()
            if final_content:
                await websocket.send_json({
                    "type": "text",
                    "content": final_content,
                    "timestamp": time.time()
                })

    except Exception as e:
        logger.error(f"Error processing OpenAI response: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "content": "Sorry, I encountered an error processing your request.",
            "error_details": str(e),  # Added error details for better debugging
            "timestamp": time.time()
        })

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            try:
                # Receive the message
                data = await websocket.receive_text()
                
                # Log received message
                logger.info(f"Received message from client {client_id}: {data}")
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "status",
                    "content": "Processing your message...",
                    "timestamp": time.time()
                })
                
                # Process the message with OpenAI
                await process_openai_response(websocket, data)
                
            except WebSocketDisconnect:
                manager.disconnect(client_id)
                break
            except Exception as e:
                logger.error(f"Error in websocket connection: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "content": f"An error occurred: {str(e)}",
                    "timestamp": time.time()
                })
    finally:
        manager.disconnect(client_id)

@app.get("/")
async def get():
    return FileResponse("static/indexmed.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    ## uvicorn main:app --reload
    
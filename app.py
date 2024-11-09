# app.py
import os
import signal
import asyncio
import speech_recognition as sr
import pyttsx3
from openai import OpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List, Dict
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
from typing import Optional





# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def read_data_files() -> List[Dict[str, str]]:
    """Read and structure knowledge base from data files."""
    texts = []
    data_dir = 'data'
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                    texts.append({
                        "filename": filename,
                        "content": file.read()
                    })
    return texts

MAX_TOKENS = 50
CONTEXT_WINDOW = 5

class BaseAgent:
    def __init__(self, llm):
        self.llm = llm
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.knowledge_base = read_data_files()
        self.chat_history = []

    def speech_to_text(self, timeout=5) -> Optional[str]:
        try:
            # Use sounddevice to record audio
            print("Listening...")
            fs = 16000  # Sampling rate
            duration = timeout  # Duration for listening
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()  # Wait for the recording to complete
            
            # Convert the recorded audio to AudioData format for recognizer
            audio_np = np.squeeze(audio)  # Remove unnecessary dimension
            audio_data = sr.AudioData(audio_np.tobytes(), fs, 2)
            
            # Recognize the speech
            text = self.recognizer.recognize_google(audio_data)
            print(f"Recognized text: {text}")
            return text
        except (sr.UnknownValueError, sr.RequestError) as e:
            print(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def text_to_speech(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def prepare_context(self, user_input: str) -> str:
        """Prepare relevant context based on user input."""
        relevant_context = []
        user_input_lower = user_input.lower()
        
        for doc in self.knowledge_base:
            content = doc["content"].lower()
            if any(word in content for word in user_input_lower.split()):
                relevant_context.append(f"From {doc['filename']}:\n{doc['content']}\n")
        
        return "\n".join(relevant_context) if relevant_context else "No specific context found."

    def update_chat_history(self, role: str, content: str):
        """Update chat history with new messages."""
        self.chat_history.append({"role": role, "content": content})
        if len(self.chat_history) > CONTEXT_WINDOW * 2:
            self.chat_history = self.chat_history[-CONTEXT_WINDOW * 2:]

class S2SAgent(BaseAgent):
    def __init__(self, llm, appointment_agent):
        super().__init__(llm)
        self.appointment_agent = appointment_agent

    def create_system_prompt(self) -> str:
        return """You are CloudJune's AI assistant. Follow these rules strictly:
1.For general greetings and personal questions:
   - Respond naturally and warmly to greetings
   - Introduce yourself as CJ, CloudJune's AI assistant
   - Be friendly and conversational while maintaining professionalism
   - For personal questions, provide engaging responses while being clear about your role as an AI
2. ONLY answer questions using information from the provided context
3. If the question cannot be answered using the provided context, respond with: "I apologize, but I can't find information about that in my knowledge base. Could you please ask something related to CloudJune's services?"
4. Never make up or infer information not present in the context
5. Keep responses concise and direct
6. Include specific references to where in the knowledge base you found the information
7.3. Interaction Style:
   - Be engaging and friendly while remaining professional
   - Use natural language and conversational tone
   - Show empathy and understanding in responses
   - Be helpful and proactive in guiding users
8. For appointment-related queries, defer to the appointment system

Current context will be provided in the user message."""

    def is_appointment_related(self, user_input: str) -> bool:
        """Check if the input is related to appointments."""
        appointment_keywords = ["appointment", "schedule", "book", "slot", "available", "time"]
        return any(keyword in user_input.lower() for keyword in appointment_keywords)

    def handle(self, user_input: str) -> str:
        """Handle user input and generate appropriate response."""
        self.update_chat_history("user", user_input)

        if self.is_appointment_related(user_input):
            response = self.appointment_agent.handle(user_input, self.chat_history)
        else:
            try:
                context = self.prepare_context(user_input)
                recent_chat = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in self.chat_history[-CONTEXT_WINDOW:]
                ])
                
                full_prompt = f"""Context Information:
{context}

Recent Conversation:
{recent_chat}

Current Question: {user_input}"""

                response = self.llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.create_system_prompt()},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=0.3
                )
                response = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error generating response: {e}")
                response = "I apologize, but I encountered an error generating a response."

        self.update_chat_history("assistant", response)
        return response

class AppointmentAgent(BaseAgent):
    def create_system_prompt(self) -> str:
        return """You are CloudJune's appointment booking assistant. Follow these rules:
1. Focus only on appointment-related queries
2. Use the provided context for available time slots and scheduling rules
3. Be clear and concise about appointment availability
4. Confirm all booking details before finalizing
5. If specific scheduling information is not in the context, ask for clarification"""

    def handle(self, user_input: str, chat_history: List[Dict[str, str]]) -> str:
        try:
            context = self.prepare_context(user_input)
            recent_chat = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in chat_history[-CONTEXT_WINDOW:]
            ])
            
            full_prompt = f"""Context Information:
{context}

Recent Conversation:
{recent_chat}

Current Appointment Request: {user_input}"""

            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating appointment response: {e}")
            return "I apologize, but I encountered an error while handling the appointment request."

load_dotenv()

# Set your OpenAI API key
oai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the client with the API key
client = OpenAI(api_key=oai_api_key)

# Initialize agents
appointment_agent = AppointmentAgent(client)
agent = S2SAgent(client, appointment_agent)

# Connection manager and WebSocket handling code remains the same...

# Global flag for controlling the listening loop
listening = False

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

def signal_handler(signum, frame):
    print("Received signal to terminate")
    global listening
    listening = False
    # Cleanup
    agent.engine.stop()
    os._exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global listening
    
    try:
        while True:
            message = await websocket.receive_text()
            
            if message == "start":
                listening = True
                await websocket.send_text("status:Listening started")
                
                while listening:
                    text = agent.speech_to_text(timeout=5)
                    
                    if text:
                        response = agent.handle(text)
                        print(f"AI Response: {response}")
                        agent.text_to_speech(response)
                        await websocket.send_text(f"response:{response}")
                    
                    # Check if a new message has been received
                    try:
                        new_message = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                        if new_message == "stop":
                            listening = False
                            agent.engine.stop()
                            await websocket.send_text("status:Listening stopped")
                            print("Listening stopped")
                            break
                    except asyncio.TimeoutError:
                        pass
                    
            elif message == "stop":
                listening = False
                agent.engine.stop()
                await websocket.send_text("status:Listening stopped")
                print("Listening stopped")
            
            else:
                # Handle text-based input
                response = agent.handle(message)
                print(f"AI Response (Text): {response}")
                await websocket.send_text(f"response:{response}")
                
    except WebSocketDisconnect:
        listening = False
        agent.engine.stop()
        manager.disconnect(websocket)
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error in WebSocket connection: {e}")
        listening = False
        agent.engine.stop()
        manager.disconnect(websocket)

@app.get("/")
async def root():
    return FileResponse("client.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Run the application with: uvicorn apptrial:app --reload

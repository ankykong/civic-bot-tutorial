import argparse
import json
import uvicorn

from voice_bot import start_bot
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo purposes or for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def start_call():
    logger.info("POST TwiML")
    return HTMLResponse(content=open("template/streams.xml").read(), media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    logger.info(f"Call data: {call_data}")
    stream_sid = call_data["start"]["streamSid"]
    logger.info("Websocket Connection Accepted")
    await start_bot(websocket, stream_sid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twilio Voice Bot Server")

    uvicorn.run(app, host="0.0.0.0", port=8765)

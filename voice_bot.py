import asyncio
import datetime
import io
import wave
import aiofiles
import os
import sys

from fastapi import WebSocket

from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.ollama import OLLamaLLMService
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator
)
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams
)

from loguru import logger
from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save.")


async def start_bot(websocket_client: WebSocket, stream_sid: str):

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            camera_out_enabled=False,
            transcription_enables=False,
            vad_enabled=True,
            vad_audio_passthrough=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.2,
                    start_secs=0.2,
                    confidence=0.4,
                )
            ),
            serializer=TwilioFrameSerializer(stream_sid)
        )
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        language="en-US",
        model="nova",
    )

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )

    llm = OLLamaLLMService(model="llama3.2")

    messages = []

    tma_in = LLMUserResponseAggregator(messages)
    tma_out = LLMAssistantResponseAggregator(messages)

    audiobuffer = AudioBufferProcessor(user_continuous_stream=False)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            tma_in,
            llm,
            tts,
            transport.output(),
            audiobuffer,
            tma_out,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
        )
    )
    SYSTEM_PROMPT = """
        You are a recruiter who is conducting an interview for a AI Engineer Position. The position will require a working on RAG (Retrieval Augmented Generation) projects.
        Your name is Allysa from Meta and you are interviewing a candidate named {name}.
        Make sure to ask these quesitons:
        1. How many years do you have in Python?
        2. Give me an example of a project you have built out.
        3. What does RAG mean?
        4. How do you measure how effective the RAG is performing?
        5. Have you worked on any open-source projects?

        Once all questions have been asked please give feedback on how the candidate did and where they can improve."""

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await audiobuffer.start_recording()
        system_prompt = SYSTEM_PROMPT.format(name="Ankur")
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": "Hello"})
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        server_name = f"server_{websocket_client.client.port}"
        await save_audio(server_name, audio, sample_rate, num_channels)

    runner = PipelineRunner()

    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(start_bot())

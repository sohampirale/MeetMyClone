#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat Cloud.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
# from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.google.llm import GoogleLLMService  # or GeminiLiveLLMService if available

from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from prompts import get_system_prompt
from pipecat.processors.transcript_processor import TranscriptProcessor 

from pathlib import Path
from PIL import Image
from pipecat.frames.frames import OutputImageRawFrame

#from pipecat.frames import VideoFrame
#from pipecat.frames.image import ImageFrame
from pipecat.frames.frames import OutputImageRawFrame,OutputAudioRawFrame
from PIL import Image
import numpy as np
import asyncio

import cv2
import subprocess

from pathlib import Path

BASE_DIR = Path(__file__).parent
IMAGE_PATH = BASE_DIR / "data" / "images" / "github_profile.png"



logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

data={
    "images":[
        {
            "filepath":"/data/images/github_profile.png",
            "purpose":"github profile of soham pirale for purpose of credibility building"
        }
    ]
}

async def send_avatar_image(pipeline, image_path: str):
    # 1) Load and convert to RGB
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # 2) Get raw bytes
    img_bytes = img.tobytes()  # RGB bytes

    # 3) Wrap in OutputImageRawFrame
    frame = OutputImageRawFrame(
        image=img_bytes,
        size=(width, height),
        format="RGB",
    )

    # 4) Push into the pipeline DOWNSTREAM so it reaches transport.output()
    await pipeline.push_frame(frame)

async def show_image(task, image_path:str):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)


    height, width, channels = arr.shape
    size = (width, height)
    data = arr.tobytes()

    #frame = VideoFrame(arr, fps=1)   # 1 FPS is enough for static image
    #frame = OutputImageRawFrame(arr)
    frame = OutputImageRawFrame(image=data,size=size,format="RGB")
    await task.queue_frames([frame])

   # while True:
        #await task.queue_frames([frame])
        #await asyncio.sleep(1)       # send every 1s to keep stream alive
        #break

async def show_video(task, video_path,audio_out:bool=True):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if fps > 0 else 1/30

    if audio_out:
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-i", str(video_path),
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        AUDIO_CHUNK_SIZE = 16000 * 2 // 20  
        # ~20ms of audio = 1600 samples * 2 bytes

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video from start
            continue

        # Convert BGR → RGB since cv2 uses BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, c = frame_rgb.shape
        size = (w, h)
        buffer = frame_rgb.tobytes()

        video_frame = OutputImageRawFrame(
            image=buffer,
            size=size,
            format="RGB"
        )
        
        await task.queue_frames([video_frame])
        
        
        # ----- AUDIO FRAME (optional) -----
        if audio_out:
            audio_bytes = ffmpeg.stdout.read(AUDIO_CHUNK_SIZE)
            if audio_bytes:
                audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000,num_channels=1)
                await task.queue_frames([audio_frame])

        await asyncio.sleep(delay)

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    
    voice_clone_id="71a7ad14-091c-4e8e-a314-022ece01c121"
    user_name='Soham Pirale'
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    transcript_processor = TranscriptProcessor() 

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=voice_clone_id, 
    )
    
    llm = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.0-flash-lite", 
        system_instruction=f"You are clone of {user_name}, and you are in a daily.co meeting You are going to talk with participants in meeting"
    )

    system_prompt=get_system_prompt(user_name)

    # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            transcript_processor.user(),
            context_aggregator.user(),  # User responses
            # llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            transcript_processor.assistant(),
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"-------Client connected")
        #image_path = "/workspaces/MeetMyClone/backend/voice-agent/server/data/images/github_profile.png"

        BASE_DIR = Path(__file__).parent
        IMAGE_PATH = BASE_DIR / "data" / "images" / "github_profile.png"

        VIDEO_PATH= BASE_DIR / "data" / "videos" / "harkirat.mp4"
        
        # asyncio.create_task(show_image(task, IMAGE_PATH))
        asyncio.create_task(show_video(task, VIDEO_PATH))
        
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        # await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=False,    # bot does NOT consume video
            video_out_enabled=True,    # bot WILL produce video
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

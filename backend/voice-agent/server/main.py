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

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.frames.frames import StartInterruptionFrame

from pyee import EventEmitter

from playwright.async_api import async_playwright
from io import BytesIO

from pipecat.frames.frames import TTSSpeakFrame


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

    frame = OutputImageRawFrame(image=data,size=size,format="RGB")
    await task.queue_frames([frame])

async def stop_showing_image(task, image_path:str):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    height, width, channels = arr.shape
    size = (width, height)
    data = arr.tobytes()

    frame = OutputImageRawFrame(image=data,size=size,format="RGB")
    await task.queue_frames([frame])




#custom processors

class GatedBufferProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._buffer = []
        self._gate_open = True
        

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._gate_open:
            #print('Allowing frames')
            await self.push_frame(frame, direction)
        else:
            #print('Buffering frames')
            # Gate is closed - buffer the frame
            self._buffer.append((frame, direction))

    async def open_gate(self):
        """Open the gate and flush all buffered frames."""
        print("Opening gate and Flushing all buffered frames")
        self._gate_open = True
        for frame, direction in self._buffer:
            await self.push_frame(frame, direction)
        self._buffer.clear()

    async def close_gate(self):
        """Close the gate to start buffering again."""
        print('Closing the gate')
        self._gate_open = False


class CustomObserver(BaseObserver):
    async def on_push_frame(self, data: FramePushed):
       # print('inside on_push_frame of Observer')
       #print(f'data : ',data)
       pass

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    
    events = EventEmitter()

    initial_state={
        "video":{
            "video_showing":True,
            "video_paused":False,
            "filepath":"",
            "video_task":None
        },
        "image":{
            "show_image":False,
            "filepath":""
        },
        "user":{
            "given_avatar":True
        }
    }
    
    def stop_showing_image():
        events.emit("image.stop_showing_image",{})

    def on_stop_showing_image(payload):
        print(f"-----------------INSIDE event handler of 'image.stop_showing_image'")
        events.emit('image.show_avatar',{})
        


    events.on("image.stop_showing_image",on_stop_showing_image)

    
    def pause_video():
        events.emit("video.pause_video",{})
    
    def on_pause_video(payload):
        initial_state['video']['video_paused']=True
        print('---------------VIDEO PAUSED-----------')
        
    events.on("video.pause_video",on_pause_video)
        
    
    def stop_video():
        events.emit('video.stop_video',{})
        
    def on_stop_video(payload):
        initial_state['video']['video_playing']=False
        initial_state['video']['video_paused']=False
        video_task = initial_state['video']['video_task']
        if video_task:
            video_task.cancel()
        initial_state['video']['video_task']=None
        
        print("-------VIDEO STOPPED-------")
        events.emit('image.show_avatar',{})
        
    events.on("video.stop_video",on_stop_video)
        
        
    def show_avatar():
        events.emit('image.show_avatar',{})
        
    def on_show_avatar(payload):
        given_avatar = initial_state["user"]["given_avatar"]
        
        if given_avatar == True:
            IMAGE_PATH = BASE_DIR / "data" / "images" / "avatar.png"
        else :
            IMAGE_PATH = BASE_DIR / "data" / "images" / "default_avatar.png"
            
        asyncio.create_task(show_image(task,IMAGE_PATH))
    
    events.on("image.show_avatar",on_show_avatar)
    
    
    def start_video_at_timestamp(start_time=0):
        events.emit('video.start_video_at_timestamp',{
            "start_time":start_time
        })
        
    def on_start_video_at_timestamp(payload):
        if "start_time" in payload:
            start_time = payload["start_time"]
        else :
            start_time=0
        
        if "filepath" in payload:
            filepath=payload['filepath']            
        else:
            filepath=initial_state['video']['filepath']
        
        if not filepath:
            print('Filepath not found cannot start video at timestamp')    
            return
        
        print(f'New start_time : {start_time}')
        
        video_task = initial_state['video']['video_task']
        
        if video_task:
            video_task.cancel()
            
        video_task=asyncio.create_task(show_video(task,filepath,start_time))
        
        print(f'------VIDEO RESTARTED AT TIMESTAMP : {start_time}-------')
         
    events.on("video.start_video_at_timestamp",on_start_video_at_timestamp)
        
    custom_observer = CustomObserver()
    
    async def show_video(task, video_path,start_time=0,audio_out:bool=True):
        
        cap = cv2.VideoCapture(str(video_path))
        video_showing=initial_state['video']['video_showing']
        video_paused = initial_state['video']['video_paused']
        
        if not cap.isOpened():
            await gated_buffer_processor.open_gate()
            
            print("Error: Cannot open video file")
            return

        if start_time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 1/30

        if audio_out:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-ss", str(start_time),
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

        while video_showing==True:
            while video_paused==True:
                video_paused=initial_state['video']['video_paused']
                await asyncio.sleep(1)
            
            ret, frame = cap.read()
            if not ret:
                #loop video after ending
                # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video from start
                # continue
                
                # when video ends stop the playback
                await gated_buffer_processor.open_gate()
                break
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
        
        
            if audio_out:
                audio_bytes = ffmpeg.stdout.read(AUDIO_CHUNK_SIZE)
                if audio_bytes:
                    audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000,num_channels=1)
                    await task.queue_frames([audio_frame])

            video_showing=initial_state['video']['video_showing']
            video_paused=initial_state['video']['video_paused']
            await asyncio.sleep(delay)

    async def show_webpage(task, url, refresh_rate=0.3):
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url)

            while True:
                # take screenshot
                img_bytes = await page.screenshot()
                
                # convert to RGB
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                arr = np.array(img)

                frame = OutputImageRawFrame(
                    image=arr.tobytes(),
                    size=(arr.shape[1], arr.shape[0]),
                    format="RGB"
                )

                await task.queue_frames([frame])
                await asyncio.sleep(refresh_rate)

    
    voice_clone_id="71a7ad14-091c-4e8e-a314-022ece01c121"
    user_name='Soham Pirale'
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    transcript_processor = TranscriptProcessor() 
    gated_buffer_processor=GatedBufferProcessor()

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
            gated_buffer_processor,
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
        observers=[RTVIObserver(rtvi),custom_observer],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"-------Client connected")
        #image_path = "/workspaces/MeetMyClone/backend/voice-agent/server/data/images/github_profile.png"

        BASE_DIR = Path(__file__).parent
        IMAGE_PATH = BASE_DIR / "data" / "images" / "github_profile.png"

        VIDEO_PATH= BASE_DIR / "data" / "videos" / "harkirat.mp4"
        
        #await gated_buffer_processor.close_gate()
        
        # asyncio.create_task(show_image(task, IMAGE_PATH))
        video_task=asyncio.create_task(show_video(task, VIDEO_PATH,5))
        initial_state["video"]['video_task']=video_task
        initial_state["video"]['filepath']=VIDEO_PATH
        
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        # await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)


    #fake stop_showing_image call
    loop = asyncio.get_event_loop()
    
    # Schedule to run once after 5 seconds
    # loop.call_later(10, lambda: stop_showing_image())

    # loop.call_later(8, lambda: pause_video())
    
    loop.call_later(10, lambda: stop_video())
    
    # loop.call_later(25, lambda: start_video_at_timestamp(5))
     
    loop.call_later(15, lambda: asyncio.create_task(show_webpage('https://google.com')))
    loop.call_later(15, lambda: asyncio.create_task(tts.push_frame(TTSSpeakFrame('This message is pushed by the STT frame'))))
     
    
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

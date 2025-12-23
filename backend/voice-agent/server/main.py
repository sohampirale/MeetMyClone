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
import cv2
import time
import numpy as np
import subprocess
import asyncio
from pipecat.frames.frames import OutputImageRawFrame, OutputAudioRawFrame
from dotenv import load_dotenv
from loguru import logger
from typing import List,Tuple
from pydantic import BaseModel, Field
print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")
from collections import deque
from pipecat.frames.frames import OutputImageRawFrame, OutputAudioRawFrame

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
import subprocess

from pathlib import Path

BASE_DIR = Path(__file__).parent

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame, 
    LLMContextFrame, 
    LLMTextFrame,      # Most common for LLM output
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame, 
    LLMMessagesFrame
)

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.frames.frames import StartInterruptionFrame

from pyee import EventEmitter

from playwright.async_api import async_playwright
from io import BytesIO

from pipecat.frames.frames import TTSSpeakFrame

from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
import cv2
import numpy as np
import subprocess
import asyncio
from io import BytesIO
from pipecat.frames.frames import OutputImageRawFrame, OutputAudioRawFrame
# import aspose.slides as slides
from pptx import Presentation
from PIL import Image
import io
from pathlib import Path
from pdf2image import convert_from_path
from strands import Agent,tool
from strands.models.litellm import LiteLLMModel
from pipecat.processors.frameworks.strands_agents import StrandsAgentsProcessor

task=None
tts_processor=None
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


class CustomProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.transcriptionFrameFound=False
        self.llmContextFrameFound=False
        self.llmMessageFrameFound=False    

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame) and  self.transcriptionFrameFound==False:
            self.transcriptionFrameFound=True
            print(f'TranscriptionFrame : {frame}')
        elif isinstance(frame, LLMContextFrame):
            self.llmContextFrameFound=True            
            print(f'LLMContextFrame : {vars(frame)}')
            print(f'context : {vars(frame.context)}')
        elif isinstance(frame, LLMMessagesFrame) and  self.llmMessageFrameFound==False:
            self.llmMessageFrameFound=True
            print(f'LLMMessage frame : {frame}')
        elif isinstance(frame,OpenAILLMContextFrame):
            print(f'OpenAILLMContextFrame frame : {vars(frame)}')
                


class CustomObserver(BaseObserver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transcriptionFrameFound=False
        self.llmContextFrameFound=False
        self.llmMessageFrameFound=False
        

    async def on_push_frame(self, frame: FramePushed, direction: FrameDirection):
        # await self.push_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and not self.transcriptionFrameFound:
            self.transcriptionFrameFound=True
            print(f'TranscriptionFrame : {frame}')
        elif isinstance(frame, LLMContextFrame):
            self.llmContextFrameFound=True            
            print(f'LLMContextFrame : {frame}')
        elif isinstance(frame, LLMMessagesFrame) and not self.llmMessageFrameFound:
            self.llmMessageFrameFound=True
            print(f'LLMMessage frame : {frame}')
        elif isinstance(frame,OpenAILLMContextFrame):
            print(f'OpenAILLMContextFrame : {OpenAILLMContextFrame}')
            

            
async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    global task
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
        "ppts":{
            "ppt_showing":True,
            "filepath":"",
            "list":[{
                'name':"sih",
                "description":"ppt of SIH",
                "goal":"Pitch SIH idea",
                "ppt_dir_path":"/data/ppts/sih/ppt2.pdf"
            }]
        },
        "user":{
            "given_avatar":True
        },
        "screen":{
            "presenting"
        }
    }

    async def convert_pngs_from_pdf(ppt_dir_path:str):
        global BASE_DIR
        output_format = "png"
        try:
            PDF_PATH = BASE_DIR / ppt_dir_path / "ppt.pdf"
            PNGS_DIR = BASE_DIR / ppt_dir_path / "pngs"
            output_dir = Path(PNGS_DIR)
            output_dir.mkdir(exist_ok=True)
            
            # Convert all PDF pages to images (non-blocking)
            images = await asyncio.to_thread(
                convert_from_path, 
                str(PDF_PATH),
                dpi=200  # Quality: 150-300 recommended
            )
            
            # Save each page as image
            for i, image in enumerate(images, 1):
                filename = f"page_{i}.{output_format}"
                image.save(output_dir / filename, output_format.upper())
                print(f"✅ Saved {filename}")
            
            print(f"✅ Total: {len(images)} pages converted!")
            return f"Converted {len(images)} pages to {output_format.upper()}"
        
        except Exception as e:
            print(f'Error show_ppt : {e}')
            return f"Error: {e}"

    png_frames: List[OutputImageRawFrame] = []

    @tool
    async def show_ppt(ppt_dir_path:str,slide_no=1):
        """Tool to present ppt"""

        global png_frames,BASE_DIR,task
        
        pngs_dir = BASE_DIR / ppt_dir_path / "pngs"

        if not pngs_dir.exists():
            print('pngs dont exist for requested ppt w')
            convert_pngs_from_pdf(ppt_dir_path)
            
        # Find all page_*.png files and sort naturally (page_1.png, page_2.png, ..., page_10.png)
        pattern = pngs_dir / "page_*.png"
        png_files = sorted(
            pngs_dir.glob("page_*.png"),
            key=lambda p: int(p.stem.split("_")[1])  # Extract number from "page_1.png" -> 1
        )
        
        if not png_files:
            print("❌ No page_*.png files found")
            return []
        
        png_frames = []
        for i, img_path in enumerate(png_files, 1):
            try:
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img)
                height, width = arr.shape[:2]  # Note: (height, width, channels)
                size = (width, height)         # OutputImageRawFrame expects (width, height)
                data = arr.tobytes()
                
                frame = OutputImageRawFrame(
                    image=data,
                    size=size,
                    format="RGB"
                )
                png_frames.append(frame)
                print(f"✅ Loaded page_{i}: {img_path.name} ({width}x{height})")
                
            except Exception as e:
                print(f"❌ Failed to load {img_path}: {e}")
        
        try:
            frame = png_frames[slide_no-1]
            await task.queue_frames([frame])
            print(f'slide no : {slide_no} pushed inot pipeline')
        except Exception as e:
            print(f'Error show_ppt : {e}')
            return f"Failed to show_ppt {e}"

        print(f"✅ Loaded {len(png_frames)} PNG frames into global list")
        return png_frames
        
    @tool
    async def change_slide(slide_no:int):
        """Tool to change slide of the ppt presentation in meeting screen
            Args : 
            slide_no : int = from 1 - n
        """
        global png_frames

        try:
            if not png_frames:
                return "No ppt is loaded into memory yet, call the tool 'show_ppt' with the ppt_dir_path"

            if slide_no> len(png_frames):
                frame = png_frames[-1]
                await task.queue_frames([frame])
                return f"Total slides of ppt are : {len(png_frames)} last slide is being presented onto the screen"

            frame = png_frames[slide_no-1]
            await task.queue_frames([frame])
            return "Requested slide is presented onto the screen"

        except Exception as e:
            print(f'Error : change_slide : {e}')
            return "Failed to present that slide_no recheck slide_no should be (1-n)"

    async def show_image(image_path:str):
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)

        height, width, channels = arr.shape
        size = (width, height)
        data = arr.tobytes()

        frame = OutputImageRawFrame(image=data,size=size,format="RGB")
        await task.queue_frames([frame])

    # async def stop_showing_image(task, image_path:str):
    #     img = Image.open(image_path).convert("RGB")
    #     arr = np.array(img)

    #     height, width, channels = arr.shape
    #     size = (width, height)
    #     data = arr.tobytes()

    #     frame = OutputImageRawFrame(image=data,size=size,format="RGB")
    #     await task.queue_frames([frame])


    
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
            
        asyncio.create_task(show_image(IMAGE_PATH))
    
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
            
        video_task=asyncio.create_task(show_video(filepath,start_time))
        
        print(f'------VIDEO RESTARTED AT TIMESTAMP : {start_time}-------')
         
         
    should_present_webpage=True

    #@tool
    def stop_webpage_present():
        """Tool to stop presenting opened webpage to the user"""
        global should_present_webpage
        should_present_webpage=False
        return "Stopped presenting webpage to the user"

    #@tool
    async def present_webpage(session_name:str):
        """Tool to present the webpage opened in browser to the user in meeting
        Args:
        session_name:str = name of the session started
        """
        while should_present_webpage==True:
            screenshot_result = browser_tool.browser({
                "action": {
                    "type": "screenshot",
                    "session_name": session_name
                }
            })
            b64_image=screenshot_result['content'][1]['data']

    events.on("video.start_video_at_timestamp",on_start_video_at_timestamp)
        
    custom_observer = CustomObserver()
    custom_processor= CustomProcessor()
    
    ####first working version
    async def show_video(task,video_path,start_time=0,audio_out:bool=True):
        global tts_processor
        frame_index=1
        #await tts_processor.started()  # ← This blocks until StartFrame received
        await asyncio.sleep(5)  # 100ms wait, other tasks run

        print(f'---------------------------INSIDE SHOW_VIDEO--------------------------------------')
        cap = cv2.VideoCapture(str(video_path))
        video_showing=initial_state['video']['video_showing']
        video_paused = initial_state['video']['video_paused']
        
        if not cap.isOpened():
            #await gated_buffer_processor.open_gate()
            
            print("Error: Cannot open video file")
            return

        if start_time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f'FPS of the video is : {fps}')
        delay = 1.0 / fps if fps > 0 else 1/30
        # delay = 1.0 / fps
        # delay = 1 / 50

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

        while video_showing==True:
            while video_paused==True:
                video_paused=initial_state['video']['video_paused']
                await asyncio.sleep(1)
            

            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, c = frame_rgb.shape
            size = (w, h)
            buffer = frame_rgb.tobytes()
            frame_index += 1
            pts = frame_index / fps
            video_frame = OutputImageRawFrame(
                image=buffer,
                size=size,
                format="RGB"
            )
            print(f'video frame : {video_frame}')
            if audio_out:
                audio_bytes = await asyncio.to_thread(
                 ffmpeg.stdout.read, AUDIO_CHUNK_SIZE
                )
                if audio_bytes:
                    audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000,num_channels=1)
                    await task.queue_frames([audio_frame,video_frame])
            else:
                await task.queue_frames([video_frame])

            

            video_showing=initial_state['video']['video_showing']
            video_paused=initial_state['video']['video_paused']
            await asyncio.sleep(delay)
        cap.release()
        if audio_out:
           ffmpeg.kill()
           ffmpeg.wait()
         
    async def show_webpage(url, refresh_rate=0.3):
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto(url,wait_until="domcontentloaded", timeout=60000)
            print('INSIDE show_webpage')

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
    global tts_processor
    tts_processor = custom_processor
    
    # llm = GoogleLLMService(
    #     api_key=os.getenv("GEMINI_API_KEY"),
    #     model="gemini-2.0-flash-lite", 
    #     system_instruction=f"You are clone of {user_name}, and you are in a daily.co meeting You are going to talk with participants in meeting"
    # )


    agent = Agent(
            name="Speaker_Agent",
            model=LiteLLMModel(
                client_args={
                    "api_key":os.getenv('OPENROUTER_API_KEY'),
                },
                model_id="openrouter/openai/gpt-4o-mini",
                params={
                    'temperature':0.5,
                    "max_tokens":1000
                }
            ),
            tools=[show_ppt,change_slide],
            system_prompt=""""You are expert speaker agent in realtime meetings (ex:Zoom,google meet, daily.co) where your job is not only to interact with user but also to present things in meeting and interact wiht user to acheive objective that has been assigned to you wisely doing everythign you can to make it as much close as possible to human touch and feel"""
    )

    strands_agent_llm = StrandsAgentsProcessor(agent=agent)

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
            strands_agent_llm,
            tts,  # TTS
            custom_processor,
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

        VIDEO_PATH= BASE_DIR / "data" / "videos" / "intro_30fps.mp4"

        
        #await gated_buffer_processor.close_gate()
        
        # asyncio.create_task(show_image( IMAGE_PATH))
        # video_task=asyncio.create_task(show_video(task,VIDEO_PATH,0))
        # initial_state["video"]['video_task']=video_task
        initial_state["video"]['filepath']=VIDEO_PATH
        
        ppt_task=asyncio.create_task(show_ppt(ppt_dir_path="data/ppts/sih",slide_no=2))
        
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
    
    #loop.call_later(15, lambda: stop_video())
    
    # loop.call_later(25, lambda: start_video_at_timestamp(5))
    
    #loop.call_later(15, lambda: asyncio.create_task(show_webpage(task,'https://google.com')))
    loop.call_later(15, lambda: asyncio.create_task(tts.push_frame(TTSSpeakFrame('This message is pushed by the STT frame'))))
    loop.call_later(15, lambda: asyncio.create_task(change_slide(4)))
    
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
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
    
    
    transport_params = {
     "daily": lambda: DailyTransport(
        room_url="https://psychsuite.daily.co/testing",  # ← Your existing room
        token=None,  # No token for open rooms
        bot_name="My Voice Bot",
        api_key=os.getenv('DAILY_API_KEY'),
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,  # Your screenshots/video
            api_key=os.getenv('DAILY_API_KEY'),
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        )
     ),
     "webrtc": lambda: TransportParams(  # Your existing WebRTC
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=False,
        video_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(),
     ),
    }
    
    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=False,    # bot does NOT consume video
            video_out_enabled=True,   
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=False,    # bot does NOT consume video
            video_out_enabled=True,    #
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()


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
print("🚀 Starting Pipecat bot...")

import os
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
    LLMMessagesFrame,
#    StartInterruptionFrame,
#    StopInterruptionFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
)
from .smart_endpointing import (
    CLASSIFIER_SYSTEM_INSTRUCTION,
    CompletenessCheck,
    OutputGate,
    StatementJudgeContextFilter,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
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
from pipecat.services.google.llm import GoogleLLMService  # or GeminiLiveLLMService if available
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from prompts import get_system_prompt
from pipecat.processors.transcript_processor import TranscriptProcessor 
from pathlib import Path
from PIL import Image
from pipecat.frames.frames import OutputImageRawFrame
from pipecat.frames.frames import OutputImageRawFrame,OutputAudioRawFrame
from PIL import Image
import numpy as np
import asyncio
import cv2
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.google import GoogleLLMService
from pipecat.services.openai import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.filters.stt_mute_filter import (
    STTMuteFilter,
    STTMuteConfig,
    STTMuteStrategy,
)
from pipecat.services.rime import RimeHttpTTSService
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.sync.event_notifier import EventNotifier
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.frames.frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
    LLMMessagesFrame,
#    StartInterruptionFrame,
#    StopInterruptionFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
)
from deepgram import LiveOptions
from loguru import logger
import time
from .smart_endpointing import (
    CLASSIFIER_SYSTEM_INSTRUCTION,
    CompletenessCheck,
    OutputGate,
    StatementJudgeContextFilter,
)
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
#from pipecat.frames.frames import StartInterruptionFrame
from pyee import EventEmitter
from playwright.async_api import async_playwright
from io import BytesIO
from pipecat.frames.frames import TTSSpeakFrame
from strands import Agent,tool
from strands.models.litellm import LiteLLMModel
from strands_tools import calculator # Import the calculator tool

model = LiteLLMModel(
    client_args={
        "api_key":os.getenv('OPENROUTER_API_KEY'),
    },
    model_id="openrouter/openai/gpt-4o-mini",
    # model_id="openrouter/google/gemini-2.0-flash-lite-001",
    # model_id="openrouter/google/gemini-2.0-flash-exp:free",
    # model_id="openrouter/google/gemini-2.0-flash-001",
    # model_id="openrouter/google/gemini-2.5-pro",
    params={
        'temperature':0.5,
        "max_tokens":1000
    },
)



@tool
def smart_background_agents(topic:str):
    """Important tool for voice agent to run background specialized powerful agents for research
    
    Use this tool mid conversation based on users interest and multiple specialized agents will run in background to collect information for you and provide you shortly

    Input : topic: describe what you want the background agents to do 
    """

    print('------------------INSIDE smart_background_agents----------------')
    print(f'topic : {topic}')
    return "background specialized agents have started"

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
        elif isinstance(frame, LLMContextFrame) and  self.llmContextFrameFound==False:
            self.llmContextFrameFound=True            
            print(f'LLMContextFrame : {frame}')
        elif isinstance(frame, LLMMessagesFrame) and  self.llmMessageFrameFound==False:
            self.llmMessageFrameFound=True
            print(f'LLMMessage frame : {frame}')


#        else:
#            print(f'frame : {frame}')
                


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
        elif isinstance(frame, LLMContextFrame) and not self.llmContextFrameFound:
            self.llmContextFrameFound=True            
            print(f'LLMContextFrame : {frame}')
        elif isinstance(frame, LLMMessagesFrame) and not self.llmMessageFrameFound:
            self.llmMessageFrameFound=True
            print(f'LLMMessage frame : {frame}')
            


            
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
    custom_processor= CustomProcessor()
    
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
    
    async def block_user_stopped_speaking(frame):
            return not isinstance(frame, UserStoppedSpeakingFrame)

    async def pass_only_llm_trigger_frames(frame):
        return (
            isinstance(frame, OpenAILLMContextFrame)
            or isinstance(frame, LLMMessagesFrame)
            or isinstance(frame, StartInterruptionFrame)
            or isinstance(frame, StopInterruptionFrame)
            or isinstance(frame, FunctionCallInProgressFrame)
            or isinstance(frame, FunctionCallResultFrame)
        )

    pipeline = Pipeline(
        [
            processor
            for processor in [
                rtvi,
                transport.input(),
                # stt_mute_filter,
                stt,  # Deepgram transcribes incoming audio
                context_aggregator.user(),
                ParallelPipeline(
                    [
                        # Branch 1: Pass everything except UserStoppedSpeakingFrame
                        FunctionFilter(filter=block_user_stopped_speaking),
                    ],
                    [
                        # Branch 2: Endpoint detection branch using Gemini for completeness
                        statement_judge_context_filter, #bunch of smart message aggregation for context for llm
                        statement_llm, #classifier YES or NO for used finished speaking or not
                        completeness_check,
                        # Use an async filter to discard branch 2's output.
                        FunctionFilter(filter=discard_all),
                    ],
                    [
                        # Branch 3: Conversation branch using Gemini for dialogue
                        FunctionFilter(filter=pass_only_llm_trigger_frames),
                        conversation_llm,
                        output_gate,
                    ],
                ),
                tts,
                user_idle,
                transport.output(),
                context_aggregator.assistant(),
            ]
            if processor is not None
        ]
    )
    
    # pipeline = Pipeline(
    #     [
    #         transport.input(),  # Transport user input
    #         rtvi,  # RTVI processor
    #         stt,
    #         transcript_processor.user(),
    #         context_aggregator.user(),  # User responses
    #         custom_processor,
    #         gated_buffer_processor,
    #         # llm,  # LLM
    #         tts,  # TTS
    #         transport.output(),  # Transport bot output
    #         transcript_processor.assistant(),
    #         context_aggregator.assistant(),  # Assistant spoken responses
    #     ]
    # )

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
    
    loop.call_later(5, lambda: stop_video())
    
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
    
    
#####################################BaseBot from other repo

class BaseBot(ABC):
    """Abstract base class for bot implementations."""

    def __init__(self, config, system_messages: Optional[List[Dict[str, str]]] = None):
        """Initialize bot with core services and pipeline components.

        Args:
            config: Application configuration.
            system_messages: Optional initial system messages for the LLM context.
        """
        self.config = config

        # Initialize STT service
        self.stt = DeepgramSTTService(
            api_key=config.deepgram_api_key, live_options=LiveOptions(model="nova-3-general")
        )

        # Initialize TTS service
        match config.tts_provider:
            case "elevenlabs":
                if not config.elevenlabs_api_key:
                    raise ValueError("ElevenLabs API key is required for ElevenLabs TTS")

                self.tts = ElevenLabsTTSService(
                    api_key=config.elevenlabs_api_key,
                    voice_id=config.elevenlabs_voice_id,
                )
            case "cartesia":
                if not config.cartesia_api_key:
                    raise ValueError("Cartesia API key is required for Cartesia TTS")

                self.tts = CartesiaTTSService(
                    api_key=config.cartesia_api_key, voice_id=config.cartesia_voice
                )
            case "deepgram":
                if not config.deepgram_api_key:
                    raise ValueError("Deepgram API key is required for Deepgram TTS")

                self.tts = DeepgramTTSService(
                    api_key=config.deepgram_api_key, voice=config.deepgram_voice
                )
            case "rime":
                if not config.rime_api_key:
                    raise ValueError("Rime API key is required for Rime TTS")

                self.tts = RimeHttpTTSService(
                    api_key=config.rime_api_key,
                    voice_id=config.rime_voice_id,
                    params=RimeHttpTTSService.InputParams(
                        reduce_latency=config.rime_reduce_latency,
                        speed_alpha=config.rime_speed_alpha,
                    ),
                )
            case _:
                raise ValueError(f"Invalid TTS provider: {config.tts_provider}")

        # Initialize LLM services
        match config.llm_provider:
            case "google":
                if not config.google_api_key:
                    raise ValueError("Google API key is required for Google LLM")

                # Main conversation LLM
                system_instruction = (
                    system_messages[0]["content"]
                    if system_messages
                    else "You are a voice assistant"
                )
                # self.conversation_llm = GoogleLLMService(
                #     api_key=config.google_api_key,
                #     model=config.google_model,
                #     params=config.google_params,
                #     system_instruction=system_instruction,
                # )
                
                self.conversation_llm= Agent(
                    model=model,
                    tools=[smart_background_agents],
                    system_prompt="You are a voice ai agent in realtime meeting with user,use tool 'smart_backgroud_agents' who will help you in the backgroud with info and task you delegate to them, DO NOT tell user about this and meantime try to answer by yourself you will know when the answer comes until them you have to handle user without that info and without letting him know that you dont have the information yet, and also in your resposnes to user add fillers like um hm or any more where appropriate to make it sound more humanistic"
                )

                self.llm = self.conversation_llm

                # Statement classifier LLM for endpoint detection
                self.statement_llm = GoogleLLMService(
                    name="StatementJudger",
                    api_key=config.google_api_key,
                    model=config.classifier_model,
                    temperature=0.0,
                    system_instruction=CLASSIFIER_SYSTEM_INSTRUCTION,
                )

            case "openai":
                if not config.openai_api_key:
                    raise ValueError("OpenAI API key is required for OpenAI LLM")

                self.conversation_llm = OpenAILLMService(
                    api_key=config.openai_api_key,
                    model=config.openai_model,
                    params=config.openai_params,
                )

                # Note: Smart endpointing currently only supports Google LLM
                raise NotImplementedError(
                    "Smart endpointing is currently only supported with Google LLM"
                )

            case _:
                raise ValueError(f"Invalid LLM provider: {config.llm_provider}")

        # Initialize context
        self.context = OpenAILLMContext(messages=system_messages)
        self.context_aggregator = self.conversation_llm.create_context_aggregator(self.context)

        # Initialize mute filter
        self.stt_mute_filter = (
            STTMuteFilter(
                stt_service=self.stt,
                config=STTMuteConfig(
                    strategies={
                        STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE,
                        STTMuteStrategy.FUNCTION_CALL,
                    }
                ),
            )
            if config.enable_stt_mute_filter
            else None
        )

        logger.debug(f"Initialised bot with config: {config}")

        # Initialize transport params
        self.transport_params = DailyParams(
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        )

        # Initialize RTVI with default config
        self.rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Initialize smart endpointing components
        self.notifier = EventNotifier()
        self.statement_judge_context_filter = StatementJudgeContextFilter(notifier=self.notifier)
        self.completeness_check = CompletenessCheck(notifier=self.notifier)
        self.output_gate = OutputGate(notifier=self.notifier, start_open=True)

        async def user_idle_notifier(frame):
            await self.notifier.notify()

        self.user_idle = UserIdleProcessor(callback=user_idle_notifier, timeout=5.0)

        # These will be set up when needed
        self.transport: Optional[DailyTransport] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None

    async def setup_transport(self, url: str, token: str):
        """Set up the transport with the given URL and token."""
        self.transport = DailyTransport(url, token, self.config.bot_name, self.transport_params)

        # Set up basic event handlers
        @self.transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            if self.task:
                await self.task.cancel()

        @self.transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await self._handle_first_participant()

        @self.transport.event_handler("on_app_message")
        async def on_app_message(transport, message, sender):
            if "message" not in message:
                return

            await self.task.queue_frames(
                [
                    UserStartedSpeakingFrame(),
                    TranscriptionFrame(
                        user_id=sender, timestamp=time.time(), text=message["message"]
                    ),
                    UserStoppedSpeakingFrame(),
                ]
            )

    def create_pipeline(self):
        """Create the processing pipeline."""
        if not self.transport:
            raise RuntimeError("Transport must be set up before creating pipeline")

        async def block_user_stopped_speaking(frame):
            return not isinstance(frame, UserStoppedSpeakingFrame)

        async def pass_only_llm_trigger_frames(frame):
            return (
                isinstance(frame, OpenAILLMContextFrame)
                or isinstance(frame, LLMMessagesFrame)
                or isinstance(frame, StartInterruptionFrame)
                or isinstance(frame, StopInterruptionFrame)
                or isinstance(frame, FunctionCallInProgressFrame)
                or isinstance(frame, FunctionCallResultFrame)
            )

        # Define an async filter that always discards frames.
        async def discard_all(frame):
            return False

        # Build pipeline with Deepgram STT at the beginning
        pipeline = Pipeline(
            [
                processor
                for processor in [
                    self.rtvi,
                    self.transport.input(),
                    self.stt_mute_filter,
                    self.stt,  # Deepgram transcribes incoming audio
                    self.context_aggregator.user(),
                    ParallelPipeline(
                        [
                            # Branch 1: Pass everything except UserStoppedSpeakingFrame
                            FunctionFilter(filter=block_user_stopped_speaking),
                        ],
                        [
                            # Branch 2: Endpoint detection branch using Gemini for completeness
                            self.statement_judge_context_filter, #bunch of smart message aggregation for context for llm
                            self.statement_llm, #classifier YES or NO for used finished speaking or not
                            self.completeness_check,
                            # Use an async filter to discard branch 2's output.
                            FunctionFilter(filter=discard_all),
                        ],
                        [
                            # Branch 3: Conversation branch using Gemini for dialogue
                            FunctionFilter(filter=pass_only_llm_trigger_frames),
                            self.conversation_llm,
                            #some custom processor who will be used for sole purpose of pushing TTSFrame into the pipeline
                            self.output_gate,
                        ],
                    ),
                    self.tts,
                    self.user_idle,
                    self.transport.output(),
                    self.context_aggregator.assistant(),
                ]
                if processor is not None
            ]
        )

        self.task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )
        self.runner = PipelineRunner()

    async def start(self):
        """Start the bot's main task."""
        if not self.runner or not self.task:
            raise RuntimeError("Bot not properly initialized. Call create_pipeline first.")
        await self.runner.run(self.task)

    async def cleanup(self):
        """Clean up resources."""
        if self.runner:
            await self.runner.stop_when_done()
        if self.transport:
            await self.transport.close()

    @abstractmethod
    async def _handle_first_participant(self):
        """Override in subclass to handle the first participant joining."""
        pass




if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

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
from pipecat.transports.daily.transport import (
    DailyTransport, DailyParams, DailyOutputTransportMessageFrame
)
from pipecat.transports.daily.utils import (
    DailyRESTHelper, DailyRoomParams, DailyRoomProperties
)
# from pipecat.transports.daily.utils import DailyRESTHelper, DailyRoomParams, DailyRoomProperties
from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper, DailyRoomParams, DailyRoomProperties
)
import time, os
import aiohttp
from pipecat.transports.daily.transport import DailyParams
from helpers import StatementJudgeContextFilter,CompletenessCheck,OutputGate
from pipecat.sync.event_notifier import EventNotifier
from prompts import CLASSIFIER_SYSTEM_INSTRUCTION
from pipecat.processors.user_idle_processor import UserIdleProcessor

task=None
tts_processor=None
transport_global=None
custom_processor_global=None
ppt_dir_path_global=None
ppt_current_slide_no =None
png_frames: List[OutputImageRawFrame] = []

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
    global task,transport_global,custom_processor_global
    transport_global=transport
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
        "images":{
            "image_showing":False,
            "list":[
                {
                    "image_path":"data/images/avatar.png",
                    "description":'avatar of soham pirale'
                },
                {
                    "image_path":"data/images/github_profile.png",
                    "description":"Github profile screenshot of soham pirale"
                }
            ]
        },
        "ppts":{
            "ppt_showing":True,
            "filepath":"",
            "list":[{
                'name':"sih",
                "description":
                    """This PPT is Team DevWise's proposal for Smart India Hackathon 2024, addressing Problem Statement 1664: developing software solutions to enhance educational infrastructure and connectivity in rural India.
                        It presents "नव शिक्षा", a comprehensive digital platform with separate portals for students, teachers, and parents, featuring AI-driven personalized learning, offline access, data analytics, resource optimization, teacher training, and secure Web3 elements.
                        The presentation covers the proposed solution, technical approach, feasibility, research references, and expected social, economic, and environmental impacts on rural education.""",
                "goal":"Pitch SIH idea",
                "ppt_dir_path":"data/ppts/sih",
                "slides_description":[
                    """Slide 1: Title PageThe slide displays the Smart India Hackathon 2024 logo and title at the top. It is labeled "TITLE PAGE" in large text. Key details include Problem Statement ID 1664, full title "Develop Software Solutions to Enhance Educational Infrastructure and Connectivity in Rural Areas", Theme: Miscellaneous, Category: Software, and Team Name: DevWise. A decorative graphic shows a brain with circuit patterns and binary code on the right side.""",
                    """Slide 2: Proposed Solution,The slide introduces the solution named "नव शिक्षा" with the team logo DevWise. It is divided into sections: Proposed Solution describes a digital platform with separate logins for students, teachers, and parents, featuring personalized learning, teacher training, parental guidance, data analytics, resource management, and AI planning. The right side explains how it addresses rural challenges through tailored portals, and lists innovations like AI personalization, community sections, AI attendance, live classes, Web3 security, and anonymous feedback.""",
                    """Slide 3: Technical ApproacThe slide details the technical workflow and tech stack. The left side shows user login branching into student, teacher, and parent paths with features like personalized content, AI recommendations, attendance tracking, progress reports, and community tools. The right side lists the tech stack: frameworks (Node.js, Flask, React, Next.js), databases (MongoDB, SQLite), languages (Python, JavaScript, HTML/CSS), cloud (AWS, Azure), AI/ML tools (Scikit-learn, PyTorch), and Web3 with Solana blockchain.""",
                    """Slide 4: Feasibility and Viability,This slide focuses on feasibility and viability. A diagram on the left shows how cloud and AI tools enable scalability, AI capabilities, and personalized learning despite implementation challenges. A central balance scale illustrates challenges like infrastructure, connectivity, community acceptance, and adoption risks. On the right, a flowchart outlines solutions: developing offline access, conducting pilot programs, and optimizing internet connectivity.""",
                    """Slide 5: Impact and Benefits,The slide outlines impact and benefits. Left side describes potential impact: improved outcomes for rural students, teacher empowerment through training, and greater parental involvement, shown in a diagram converging to positive rural education transformation. Right side lists benefits: social (bridging urban-rural gap), economic (better job opportunities and growth), environmental (reduced paper use), illustrated with icons and arrows leading to positive impacts.""",
                    """Slide 6: Research and References,This slide presents research backing the solution. It covers the condition of rural education citing ASER 2022 report on low literacy. Technology in rural education references UNESCO ICT report and World Economic Forum on digital initiatives. Pilot projects highlight EkStep Foundation's open platforms. Benefits section cites Brookings report on e-learning and J-PAL study on low-cost tools improving literacy and numeracy, with hyperlinks provided for each source."""
                ]
            }]
        },
        "links":{
            "list":[
                {
                    "link":'https://github.com/sohampirale',
                    "description":'Github url of soham pirale',
                    "goal":'Send this to build credibility when project building is related'
                },
                {
                    "link":'https://github.com/sohampirale/n8n_clone',
                    "description":'n8n clone project by soham pirale',
                    "goal":'Full stack n8n clone , send/show this to build strong credibility about ai and n8n like projects'
                },
                 {
                    "link":'https://github.com/sohampirale/DockHostV2',
                    "description":'DockHost SSH enabled project by soham pirale',
                    "goal":'Docker and SSH container enabling project by soham pirale, send/show this to build strong credibility about Devops and Docker related projects'
                }
            ]
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


    @tool
    async def show_ppt(ppt_dir_path:str,slide_no=1,tts_message:str=""):
        """Tool to present ppt
            Args:
            ppt_dir_path:str = exact same dir path given in context messages about ppts
            slide_no:int(1 to n) (optional) 1 by default
            tts_message:str (optional) = Temporary filler message for user in realtime meeting (ex: Just a moment, or anything that you find appripriate ), this tts_message will immediately be converted to audio and sent to user to show low latency, (empty string by default)
        """

        global png_frames,BASE_DIR,task,ppt_dir_path_global,ppt_current_slide_no
        
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
            return "Pngs not availaible for this ppt"
        
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
        except Exception as e:
            print(f'Error show_ppt : {e}')
            return f"Failed to show_ppt {e}"

        ppt_dir_path_global=Path(ppt_dir_path).resolve()
        ppt_current_slide_no=slide_no
        print(f'slide no : {slide_no} pushed inot pipeline')

        slide_description=""
        all_ppt_lists=initial_state['ppts']['list']

        for ppt in all_ppt_lists:
            if ppt['ppt_dir_path']==ppt_dir_path_global:
                slide_description=ppt['slides_description'][slide_no-1]

        screen_status={
            "role":'system',
            "showing_image":False,
            "showing_video":False,
            "showing_ppt":True,
            "slide_no":ppt_current_slide_no,
            "slide_description":slide_description ,
            "total_slides_in_this_ppt":len(png_frames)
        }
        print(f"✅ Loaded {len(png_frames)} PNG frames into global list")
        return png_frames
        
    @tool
    async def change_slide(slide_no:int,tts_message:str=""):
        """Tool to change slide of the ppt presentation in meeting screen
            Args : 
            slide_no : int = from 1 - n
            tts_message:str (optional) = Temporary filler message for user in realtime meeting (ex: Just a moment, or anything that you find appripriate ), this tts_message will immediately be converted to audio and sent to user to show low latency
        """
        global png_frames,transport_global,custom_processor_global,ppt_dir_path_global,ppt_current_slide_no
        print(f'slide no : {slide_no}')
        try:
            if not png_frames:
                return "No ppt is loaded into memory yet, call the tool 'show_ppt' with the ppt_dir_path from context given to you"

            slide_description=""

            if slide_no> len(png_frames):
                frame = png_frames[-1]
                # await task.queue_frames([frame])
                await custom_processor_global.push_frame(frame)
                ppt_current_slide_no=len(png_frames)

                all_ppt_lists=initial_state['ppts']['list']
                for ppt in all_ppt_lists:
                    if Path(ppt['ppt_dir_path'])==ppt_dir_path_global:
                        slide_description=ppt['slides_description'][-1]

                return f"Total slides of ppt are : {len(png_frames)} last slide is being presented onto the screen, description of last slide is : {slide_description}"

            frame = png_frames[slide_no-1]

            await custom_processor_global.push_frame(frame)
            all_ppt_lists=initial_state['ppts']['list']
            ppt_current_slide_no=slide_no

            for ppt in all_ppt_lists:
                if Path(ppt['ppt_dir_path']).resolve()==ppt_dir_path_global:
                    slide_description=ppt['slides_description'][slide_no-1]
                    break

            return f"Requested slide is presented onto the screen, description of that slide is : {slide_description}"

        except Exception as e:
            print(f'Error : change_slide : {e}')
            return f"Failed to present that slide_no recheck slide_no )"

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
         
    @tool
    async def send_message(message:str):
        """Tool to send message in the realtime meeting to the user
        Args:
        message:str = Exact message will be sent to the user in chat
        """
        print('INSIDE send_message')
        try:
            await transport.send_prebuilt_chat_message(
                message,
                user_name="MeetingBot"
            )
            return "message sent"
        except Exception as e:
            print(f'Error send_message : {e}')
            return f"Error : {e}"
         
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

    custom_processor_global = custom_processor
    
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
            tools=[show_ppt,change_slide,send_message],
            system_prompt=""""You are expert speaker agent in realtime meetings (ex:Zoom,google meet, daily.co) where your job is not only to interact with user but also to present things in meeting and interact wiht user to acheive objective that has been assigned to you wisely doing everythign you can to make it as much close as possible to human touch and feel"""
    )

    strands_agent_llm = StrandsAgentsProcessor(agent=agent)

    system_prompt=get_system_prompt(user_name)

    # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    all_ppt_lists=initial_state['ppts']["list"]
    ppts_context=[]

    for ppt in all_ppt_lists:
        temp ={
            "ppt_dir_path":ppt["ppt_dir_path"],
            "description":ppt["description"],
            "goal":ppt["goal"]
        }
        ppts_context.append(temp)

    images_context=[]
    videos_context=[]
    links_context=initial_state['links']['list']
        
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role":'system',
            "content":"""In this obj ,all availaible images are mentioned in short""",
            "images_context":images_context
        },
        {
            "role":'system',
            "content":"""In this obj ,all availaible videos are mentioned in short""",
            "videos_context":videos_context
        },
        {
            #ppts
            "role":'system',
            "content":"""In this obj ,all availaible ppts are mentioned in short""",
            "ppts_context":ppts_context
        },
        {
            "role":"system",
            "content":"All links and their goal and descriptions will be attached here, use them wisely and send them in chat whenever appropriate proactively",
            "links":links_context
        },
        {
            "role":'system',
            "content":'Current status of presenting/displaying images/videos/ppt will be mentioned here, currently nothing is being displyed'
        }
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

    ####working with turn detection enabled pipeline

    system_instruction="""You are great voice ai agent"""

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

    async def discard_all(frame):
        return False


    notifier=EventNotifier()

    statement_llm = GoogleLLMService(
        name="StatementJudger",
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.0-flash-lite",
        temperature=0.0,
        system_instruction=CLASSIFIER_SYSTEM_INSTRUCTION,
    )

    statement_judge_context_filter=StatementJudgeContextFilter(notifier)
    completeness_check=CompletenessCheck(notifier)

    conversation_llm =StrandsAgentsProcessor(agent=agent)
    output_gate= OutputGate(notifier=notifier,start_open=True)

    async def user_idle_notifier(frame):
            await notifier.notify()

    user_idle = UserIdleProcessor(callback=user_idle_notifier, timeout=5.0)

    pipeline = Pipeline(
        [
            
            rtvi,
            transport.input(),
            # stt_mute_filter,
            stt,  # Deepgram transcribes incoming audio
            context_aggregator.user(),
            ParallelPipeline(
                [
                    FunctionFilter(filter=block_user_stopped_speaking),
                ],
                [
                    statement_judge_context_filter,
                    statement_llm,
                    completeness_check,
                    FunctionFilter(filter=discard_all),
                ],
                [
                    FunctionFilter(filter=pass_only_llm_trigger_frames),
                    conversation_llm,
                    output_gate,
                ],
            ),
            tts,
            # user_idle, #TODO use this later if needed
            transport.output(),
            context_aggregator.assistant(),
            
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=60,  # Cancel if no activity for 60 seconds
        cancel_on_idle_timeout=True,  #
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

    @task.event_handler("on_idle_timeout")
    async def on_idle_timeout(task):
        logger.info("No user joined - cleaning up")
        await task.cancel()

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
    loop.call_later(15, lambda: asyncio.create_task(send_message('hey there')))

    
    await runner.run(task)

async def create_chat_room() -> tuple[str, str]:
    """Create a Daily room with chat enabled."""
    async with aiohttp.ClientSession() as session:
        helper = DailyRESTHelper(
            daily_api_key=os.getenv("DAILY_API_KEY"),
            aiohttp_session=session
        )
        
        # Create room with custom properties
        room = await helper.create_room(DailyRoomParams(
            properties=DailyRoomProperties(
                exp=time.time() + 3600,  # 1 hour expiry
                enable_chat=True
            )
        ))
        
        # Generate token
        token = await helper.get_token(room.url, expiry_time=3600)
        return room.url,token
    
    
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
            video_in_enabled=False,   
            video_out_enabled=True,   
            api_key=os.getenv('DAILY_API_KEY'),
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

    #TODO : add room and token creation to different endpoint - prod

    # room_url, token = await create_chat_room()
    # transport = DailyTransport(room_url, token, "Bot", DailyParams(
    #                 audio_in_enabled=True,
    #                 audio_out_enabled=True,
    #                 video_out_enabled=True,  
    #                 # api_key=os.getenv('DAILY_API_KEY'),
    #                 vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    #                 turn_analyzer=LocalSmartTurnAnalyzerV3(),
    #             )
    # )


    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()



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
#BASE_DIR = Path(__file__).parent
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame, 
    LLMContextFrame, 
    LLMTextFrame,      # Most common for LLM output
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame, 
    LLMMessagesFrame
)
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
from pipecat.pipeline.parallel_pipeline import ParallelPipeline

from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.frames.frames import Frame
#from pipecat.frames.frame_direction import FrameDirection  # Correct location
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    LLMMessagesFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
    StartInterruptionFrame,
    BotInterruptionFrame,
    LLMContextFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frameworks.strands_agents import StrandsAgentsProcessor

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

from strands import Agent,tool
from strands.models.litellm import LiteLLMModel
from strands_tools import calculator # Import the calculator tool


@tool
def smart_background_agents(topic:str):
    """Important tool for voice agent to run background specialized powerful agents for research
    
    Use this tool mid conversation based on users interest and multiple specialized agents will run in background to collect information for you and provide you shortly

    Input : topic: describe what you want the background agents to do 
    """

    print('------------------INSIDE smart_background_agents----------------')
    print(f'topic : {topic}')
    return "background specialized agents have started"

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

agent = Agent(
    model=model,
    tools=[calculator,smart_background_agents],
    system_prompt="You are a voice ai agent in realtime meeting with user,use tool 'smart_backgroud_agents' who will help you in the backgroud with info and task you delegate to them, DO NOT tell user about this and meantime try to answer by yourself you will know when the answer comes until them you have to handle user without that info and without letting him know that you dont have the information yet, and also in your resposnes to user add fillers like um hm or any more where appropriate to make it sound more humanistic"
)


messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant. When the conversation starts say 'Hey there Soham'",
        }
]

context = LLMContext(messages)
context_aggregator = LLMContextAggregatorPair(context)


def dump_frame(frame):
    print(f"\n=== {frame.__class__.__name__} ===")
    for attr, value in vars(frame).items():
        print(f"{attr}: {value}")

class AWSStrandsProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.seen_llm_messages = False
        self.seen_user_stopped = False
        self.seen_transcription = False
        self.seen_start_interruption = False
        self.seen_stop_interruption = False
        self.seen_llm_context = False
        
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)
        if isinstance(frame, LLMMessagesFrame) and not self.seen_llm_messages:
            print("LLMMessagesFrame (LLM INPUT, not trigger)")
            self.seen_llm_messages = True
            dump_frame(frame)


        elif isinstance(frame, UserStoppedSpeakingFrame) and not self.seen_user_stopped:
            print("UserStoppedSpeakingFrame (TURN END signal)")
            self.seen_user_stopped = True
            dump_frame(frame)

        elif isinstance(frame, TranscriptionFrame) and not self.seen_transcription:
            print("TranscriptionFrame (PARTIAL / INTERMEDIATE ASR)")
            self.seen_transcription = True
            dump_frame(frame)

        elif isinstance(frame, BotInterruptionFrame) and not self.seen_start_interruption:
            print("BotInterruptionFrame (INTERRUPTION BOT control)")
            self.seen_start_interruption = True
            dump_frame(frame)

        elif isinstance(frame, LLMContextFrame) and not self.seen_llm_context:
            #gives context not for triggering
            print(f'context at the moment is ')
            dump_frame(context)
            dump_frame(frame)
            #print("LLMContextFrame ",frame)
            #self.seen_llm_context = True

        elif isinstance(frame,OpenAILLMContextFrame) and not self.seen_llm_context:
            #gives context not for triggering
            dump_frame(frame)
            #print("LLMContextFrame ",frame)
            #self.seen_llm_context = True


        



async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    print('--------INSIDE run_bot()')

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )
    
    aws_strands=AWSStrandsProcessor()
    strands_agent = StrandsAgentsProcessor(agent=agent)
    # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    llm = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.0-flash",  # or another Gemini model variant
        system_instruction="You are a helpful AI assistant. Keep responses brief."
    )


    #messages = [
    #    {
    #        "role": "system",
    #        "content": "You are a friendly AI assistant. When the conversation starts say 'Hey there Soham'",
    #    },
    #]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            aws_strands,       
            strands_agent,
            tts,  # TTS
            transport.output(),  # Transport bot output
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
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""
    print('---------INSIDE bot()')
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
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()


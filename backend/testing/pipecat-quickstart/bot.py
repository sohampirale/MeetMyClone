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
import asyncio
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
from pipecat.frames.frames import Frame,LLMTextFrame

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
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from pipecat.frames.frames import OutputImageRawFrame  # adjust path if needed

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frameworks.strands_agents import StrandsAgentsProcessor
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=3)

task=None
tts_processor=None
llm_processor=None
logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)
from typing import Dict,List,Any
from strands import Agent,tool
from strands.models.litellm import LiteLLMModel
from strands_tools import calculator # Import the calculator tool
from typing import List,Tuple
from pydantic import BaseModel, Field
from strands_tools.browser import LocalChromiumBrowser
from strands_tools.browser.models import BrowserInput
from pathlib import Path
BASE_DIR = Path(__file__).parent


class TaskJob(BaseModel):
    agent: str = Field(..., description="Name of the agent")
    task: str = Field(..., description="Specific task for this agent")

class AssignTasksInput(BaseModel):
    jobs: List[TaskJob] = Field(..., description="List of tasks to assign")


messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant. When the conversation starts say 'Hey there Soham'",
        },
        [],
        {},
        {},
        {}
]

context = LLMContext(messages)
context_aggregator = LLMContextAggregatorPair(context)

all_availaible_agents={
    "CreativeAgent":{
            'description':'Specialized agent in creative thinking',
            'agent':Agent(
                name="CreativeAgent",
                model=LiteLLMModel(
                    client_args={
                        "api_key":os.getenv('OPENROUTER_API_KEY'),
                    },
                    model_id="openrouter/openai/gpt-4o-mini",
                    params={
                        'temperature':0.65,
                        "max_tokens":1000
                    }
                ),
                tools=[],
                system_prompt="You are a creative agent helping your coworker Speaker agents who is interacting with the user in realtime"
        )
    },
    "SalesAgent":{
        "description":"Specialized agent in sales",
        "agent":Agent(
            name="SalesAgent",
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
            tools=[],
            system_prompt="You are an expert sales agent helping your coworker Speaker agents who is interacting with the user in realtime in an meeting"
        )
    },
    "GithubAgent":{
        "description":"Specialized agent in Github knowledge and Github realtime data",
        "agent":Agent(
                name="GithubAgent",
                model=LiteLLMModel(
                    client_args={
                        "api_key":os.getenv('OPENROUTER_API_KEY'),
                    },
                    model_id="openrouter/openai/gpt-4o-mini",
                    params={
                        'temperature':0.25,
                        "max_tokens":1000
                    }
                ),
                tools=[],
                system_prompt="You are a expert gitub agent who is capable finding repos on github and knows everything about github"
        )
    },
    "CriticAgent":{
        "description":"Specialized agent in reasoning and evaluating things as a safety measure",
        "agent":Agent(
            name="CriticAgent",
            model=LiteLLMModel(
                client_args={
                    "api_key":os.getenv('OPENROUTER_API_KEY'),
                },
                model_id="openrouter/openai/gpt-4o-mini",
                params={
                    'temperature':0.1,
                    "max_tokens":1000
                }
            ),
            tools=[],
            system_prompt="You are a critic agent who critics things constructively helping your coworker Speaker agent who is interacting with the user in realtime"
    )},
    "IdeasAgent":{
        "description":"Specialized agent in generating ideas",
        "agent":Agent(
            name="IdeasAgent",
            model=LiteLLMModel(
                client_args={
                    "api_key":os.getenv('OPENROUTER_API_KEY'),
                },
                model_id="openrouter/openai/gpt-4o-mini",
                params={
                    'temperature':0.65,
                    "max_tokens":1000
                }
            ),
            tools=[],
            system_prompt="You are an expert in generating ideas and you are helping your coworker Speaker agent who is interacting with the user in realtime"
    )},
}

@tool
def create_new_agent(agent_name:str,agent_descirpion:str):
    """Tool for creating new agent from scratch
    Args:
        agent_name:str = name for this new agent (IMP : It should be unique and not same as any agent!)
        agent_description:str = clearly explain what should this agent is capable of doing (ex : Specialized agent in 'xyz')
        ex Sales agents : Expert in sales for general public 
    """
    
    system_prompt_generator_agent=Agent(
            model=LiteLLMModel(
                client_args={
                    "api_key":os.getenv('OPENROUTER_API_KEY'),
                },
                model_id="openrouter/openai/gpt-4o-mini",
                params={
                    'temperature':0.4,
                    "max_tokens":1000
                }
            ),
            tools=[],
            system_prompt="You are expert in generating system prompts for creating new specialized agents for that specific task, output with system prompts for requested agent, No explainations!"
    )
    
    new_agent_system_prompt=system_prompt_generator_agent(agent_descirpion)
    
    new_agent=Agent(
            name=agent_name,
            model=LiteLLMModel(
                client_args={
                    "api_key":os.getenv('OPENROUTER_API_KEY'),
                },
                model_id="openrouter/openai/gpt-4o-mini",
                params={
                    'temperature':0.4,
                    "max_tokens":1000
                }
            ),
            tools=[],
            system_prompt=new_agent_system_prompt
    )
    
    all_availaible_agents[agent_name]={
        "description":agent_descirpion,
        "agent":new_agent
    }
    
    return "Requested agent created successfully"

async def run_agent_job(agent: any, task: str):
    output = agent(task)
    print(f'\n-----Job finished of {agent.name} Agent------------\n')
    context.messages[1].append({
        "agent":agent.name or "background_agent",
        "response":str(output)
    })

browser_tool = LocalChromiumBrowser()

# @tool
# async def set_page_instance():
#     """Extract REAL page from session"""
#     global browser_tool, cnt, current_browser_page

#     session_name = "google-session"

#     print("\n🔍 CHECKING SESSIONS...")
#     print(f"Sessions: {list(browser_tool._sessions.keys())}")
#     print(f"browser_tool._sessions: {browser_tool._sessions}")

#     if session_name in browser_tool._sessions:
#         session = browser_tool._sessions[session_name]
#         page = session.page

#         print(f"✅ SUCCESS! PAGE FOUND: {page}")
#         print(f"✅ URL: {page.url}")
#         print(f"✅ Type: {type(page)}")

#     print("❌ No google-session yet")
#     return "page instance set successfully"

should_present_webpage=True

@tool
def stop_webpage_present():
    """Tool to stop presenting opened webpage to the user"""
    global should_present_webpage
    should_present_webpage=False
    return "Stopped presenting webpage to the user"

async def _async_present_webpage(session_name:str):
    global should_present_webpage,task,executor
    loop = asyncio.get_event_loop()

    while should_present_webpage==True:
        try:
            #screenshot_result = browser_tool.browser({
            #    "action": {
            #        "type": "screenshot",
            #        "session_name": session_name
            #    }
            #})
            #screenshot_result = await loop.run_in_executor(
            #    executor,
            #    lambda: browser_tool.browser({
            #        "action": {
            #            "type": "screenshot",
            #            "session_name": session_name
            #        }
            #    })
            #)

            #with thread same contect copied
            screenshot_result = await asyncio.to_thread(  # ✅ NON-BLOCKING
              browser_tool.browser,
              {"action": {"type": "screenshot", "session_name": session_name}}
            )

            
            #print(f'screenshot_result : {screenshot_result}')
            if screenshot_result.get("status") != "success":
                await asyncio.sleep(0.5)
                continue

            text_msg = screenshot_result['content'][0]['text']
            filepath = text_msg.replace('Screenshot saved as ', '').strip()
            
                # 2) Extract base64 image from content
            #b64_image = screenshot_result["content"][1]["data"]

            ## 3) Decode base64 → bytes → PIL → numpy
            #img_bytes = base64.b64decode(b64_image)
            #img = Image.open(BytesIO(img_bytes)).convert("RGB")
            #TODO use image path to psuh down
            FINAL_IMAGE_PATH = BASE_DIR / filepath
            print(f'FINAL_IMAGE_PATH : {FINAL_IMAGE_PATH}')
            img = Image.open(FINAL_IMAGE_PATH).convert("RGB")
            arr = np.array(img)
            h, w, c = arr.shape
            size = (w, h)
            data = arr.tobytes()

            # 4) Create Pipecat frame and queue as “video” frame
            frame = OutputImageRawFrame(
                image=data,
                size=size,
                format="RGB",
            )
            print(f'Pushed image frame : {frame}')
            global tts_processor
            await tts_processor.push_frame(frame)
            #await task.queue_frames([frame])
        except Exception as e:
            print(f'Error in _async_present_webpage : {e}')

        # 5) Small delay to control “FPS”
        await asyncio.sleep(0.5)

@tool
async def present_webpage(session_name:str):
    """Tool to present the webpage opened in browser to the user in meeting
    Args:
    session_name:str = name of the session started
    """
    print('-----Inside present webpage')
    asyncio.create_task(_async_present_webpage(session_name))
    
    return "webpage started presenting successfully"

@tool
async def assign_tasks(jobs:AssignTasksInput):
    """
    Assign independent tasks to multiple agents for parallel execution.

    Each job is executed in isolation. Agents do NOT collaborate or share context.

    Args:
        jobs (List[Tuple[str, str]]):
            A list of (agent_name, task_description) pairs.

            - agent_name (str):
                Must exactly match one of the available agent identifiers.
                Case-sensitive. No aliasing or inference is performed.

            - task_description (str):
                A clear, self-contained task for the agent to complete independently.

    Behavior:
        - All jobs are dispatched in parallel.
        - No coordination, communication, or result-sharing occurs between agents.
        - This tool does not return results; downstream systems must collect outputs separately.
    """
    print(f'---------INSIDE assign_tasks()--------------')
    print(f'jobs : {jobs}')
    #TODO : add error handlign and log error
    try:
        jobs = jobs["jobs"]
        for obj in jobs:
            agent_name = obj["agent"]
            task = obj["task"]
            if agent_name in all_availaible_agents and "agent" in all_availaible_agents[agent_name]:
                agent=all_availaible_agents[agent_name]["agent"]
                asyncio.create_task(run_agent_job(agent,task))
                print(f'Assigned task : {task} to agent {agent.name}')
            else:
                print(f'agent_name not found in all_availaible_agents : {agent_name}')
    except Exception as e:
        print(f'Errro in assing_task() : {e}')

    print('Returning the statement')
    return "All Agents have started working on given jobs"

async def run_agent_blocking(agent, prompt: str):
    return await asyncio.to_thread(agent, prompt)

async def _run_background_agents(task: str):
    agents_creator_agent = Agent(
            model=LiteLLMModel(
                    client_args={
                        "api_key":os.getenv('OPENROUTER_API_KEY'),
                    },
                    model_id="openrouter/openai/gpt-4o-mini",
                    params={
                        'temperature':0.1,
                        "max_tokens":1000
                    }
                ),
            tools=[create_new_agent],
            system_prompt="You are expert at decision making and deciding which exact agents are necessary for a job,Your job is to understand the requirement or task given to us and depending on it we are going to spawn multiple agents to work on that task, you have to decide based on currently availaible agents and their speciality whether to create new agents whenever necessary and work with currently availaible once,use the tool create_new_agent and that will create that specialized agent in backgroud, all agents are going to run in paralle and not communictae with each main purpose for building them is to getr commulitive specialized responses for that task, IMP : Requested agents will be created 100% even though they might not be present in all availaible agents, DO NOT request for same agent twice!"
    )
    
    #agents_utilization_response=agents_creator_agent(f'Currently availaible agents are : {all_availaible_agents} and requirement or task description is : {task}, your end response should clearly explain which agents to use (that are already existsing) and all the agents you created using create_new_agent tool and what to ask these agent to do')
    
    orchestrator_agent = Agent(
        model=LiteLLMModel(
                client_args={
                    "api_key":os.getenv('OPENROUTER_API_KEY'),
                },
                model_id="openrouter/openai/gpt-4o-mini",
                params={
                    'temperature':0.1,
                    "max_tokens":1000
                }
            ),
        tools=[assign_tasks],
        system_prompt="You are an expert agent orchestrator your main job is to understand requirement or task thats asked and having all availaible agents assign specific things to each agent to do in parallel non collaborative way that we will collect output from each agent at the end,You will be given all availaible agents with their description , use tool 'assign_tasks' to assign taks to agents, your final output doesnt matter but how you use the tool 'assign_tasks' does, do not output anythign at the end, IMP : DO NOT use the 'assign_tasks' more than once"
    )
#    orchestrator_agent(f'Task is {task} {agents_utilization_response} ,all availaible agent : {all_availaible_agents}')
#    orchestrator_agent(f'Task is {task} ,all availaible agent : {all_availaible_agents}')

    await run_agent_blocking(
        orchestrator_agent,
        f"Task is {task}. Agents: {all_availaible_agents}"
    )

# @tool
# async def init_browser_session():
#     """Initialize AND verify session"""
#     global browser_tool
#     await browser_tool.init_session("main")
    
#     # CHECK if session was created
#     print(f"Sessions after init: {browser_tool._sessions}")
#     if "main" in browser_tool._sessions:
#         print("✅ 'main' session created!")
#     else:
#         print("❌ Session NOT created!")
    
#     return "Session initialized"

@tool
async def smart_background_agents(task:str):
    """Important helper tool to run background specialized powerful agents for research
    
    Use this tool mid conversation based on users interest and multiple specialized agents will run in background to collect information for you and provide you shortly

    Input : task: describe what you want the background agents to do 
    """
    
    print('------------------INSIDE smart_background_agents---------------- ')
    print(f'task : {task}')
        
    asyncio.create_task(_run_background_agents(task))

    
    return "background specialized agents have started working in backgroud and will update your system prompt with their knowledge in next inference of you"

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
async def browser(browser_input: BrowserInput) -> Dict[str, Any]:
    """
    Browser automation tool for web scraping, testing, and automation tasks.

    This tool provides comprehensive browser automation capabilities using Playwright
    with support for multiple browser engines. It offers session management, tab control,
    page interactions, content extraction, and advanced automation features.

    Usage with Strands Agent:
    ```
    from strands import Agent
    from strands_tools.browser import Browser

    # Create the browser tool
    browser = Browser()
    agent = Agent(tools=[browser.browser])

    # Initialize a session
    agent.tool.browser(
        browser_input={
            "action": {
                "type": "init_session",
                "description": "Example session",
                "session_name": "example-session"
            }
        }
    )

    # Navigate to a page
    agent.tool.browser(
        browser_input={
            "action": {
                "type": "navigate",
                "url": "https://example.com",
                "session_name": "example-session"
            }
        }
    )

    # Close the browser
    agent.tool.browser(
        browser_input={
            "action": {
                "type": "close",
                "session_name": "example-session"
            }
        }
    )
    ```

    Args:
        browser_input: Structured input containing the action to perform.

    Returns:
        Dict containing execution results (returns immediately while action executes in background).
    """
    # Fire & forget - execute in background
    print(f'browser_input : {browser_input}')
    asyncio.create_task(
        _execute_browser_background(browser_input)
    )
    
    # Return immediately to LLM
    action_type = "browser action"
    try:
        if isinstance(browser_input, dict):
            action_type = browser_input.get("action", {}).get("type", "browser action")
        else:
            action_type = browser_input.action.type
    except:
        pass
    
    return {
        "status": "started",
        "content": [{
            "text": f"✅ {action_type.title()} started in background. This may take a few seconds to complete."
        }]
    }


async def _execute_browser_background(browser_input: BrowserInput):
    """Execute browser action in background on separate thread."""
    global browser_tool
    try:
        result = await asyncio.to_thread(
            browser_tool.browser,
            browser_input
        )
        print(f'✅ Browser action completed: {result}')
    except Exception as e:
        print(f'❌ Browser action error: {e}')


# tools=[browser_tool.browser,init_browser_session,set_page_instance,smart_background_agents,present_webpage,stop_webpage_present]
tools=[browser_tool.browser,present_webpage,stop_webpage_present]


browser_agent = Agent(
    model=model,
    tools=tools,
    system_prompt="You are an expert agent in using Browser Tools, Your job is to understand query given to you and use tools wisely to present it in the realtime meeting"
)

async def _invoke_browser_agent_async(prompt: str):
    """Invoke agent on separate thread."""
    global browser_agent
    print('Inside _invoke_browser_agent_async')
    try:
        # Run synchronous agent.invoke() on thread
        result = await asyncio.to_thread(
            browser_agent,  
            prompt
        )
        print(f'✅ Agent result: {result}')
    except Exception as e:
        print(f'❌ _invoke_browser_agent_async error: {e}')

@tool
async def assign_task_to_browser_agent(task:str):
    """Tool to assign browser operations and things to the specialized agent
    Args:
    task:str = task description to perform with browser tools
    """
    
    print(f'Inside assign_task_to_browser_agent task : {task}')
    
    asyncio.create_task(_invoke_browser_agent_async(task))
    return f"✅ Task assigned to the browser_agent, continue interacting with user until browser_agent finishes the task"

@tool
async def say_to_user(text:str):
    """Tool to send transcript for TTS in realtime meeting
    Args:
    text:str = This string will be conevrted to audio and sent to user in meeting
    """
    global llm_processor
    await llm_processor.push_frame(LLMTextFrame(text))
    return "request text sent to user by TTS"
    
    
    

agent = Agent(
    model=model,
    tools=[assign_task_to_browser_agent,say_to_user],
    # system_prompt="You are a voice ai agent in realtime meeting with user,use tool 'smart_backgroud_agents' who will help you in the backgroud with info and task you delegate to them,their findings will be available shortly ,tell user that you have agents working in background and meantime assist user by yourself with whatever knowledge you have! dont give too much technical internal details as well , and also in your resposnes to user add fillers like um hm or any more where appropriate to make it sound more humanistic, be more of real human in voice and not agent,call the tool 'smart_background_agent' before responding to user and keep interacting onwards, Do not wait for information to come you have to continue interacting with user with whatever you know, IMP : after successfull browser opening call the tool 'present_webpage'"
    # system_prompt="You are a voice ai agent in realtime meeting with user, you are expert in using browser with tools atatched to you, everytime you open a browser use tool 'present_webpage' and 'stop_webpage_present' after user says close webpage"
    # system_prompt="You are an expert voice ai agent in talking with users in realtime meeting, when asked for any browser related operation assign that task to the 'assign_task_to_browser_agent' tool and interact with the user with 'say_to_user' tool and not your output"    
    system_prompt="""
        You are an expert voice AI agent for real-time meetings.

        RULES:
        1. say_to_user("message") → ALL speech ONLY
        2. assign_task_to_browser_agent("simple instruction") → Browser tasks ONLY

        PATTERN: say → assign → say → END (NO TEXT OUTPUT)

        EXAMPLES:
        "open google" → say_to_user(anymessage) + assign_task_to_browser_agent(task) → return empty string ""
    """

)


def dump_frame(frame):
    print(f"\n=== {frame.__class__.__name__} ===")
    for attr, value in vars(frame).items():
        print(f"{attr}: {value}")

#custom processor for testing
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
    global task
    logger.info(f"Starting bot")
    print('--------INSIDE run_bot()')

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )
    global tts_processor
    tts_processor=tts
    aws_strands=AWSStrandsProcessor()
    
    global llm_processor
    strands_agent_llm = StrandsAgentsProcessor(agent=agent)
    llm_processor=strands_agent_llm
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
            strands_agent_llm,
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
            video_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()


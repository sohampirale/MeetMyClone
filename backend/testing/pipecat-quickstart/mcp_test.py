import os
import asyncio
from dotenv import load_dotenv
from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent,tool
from strands.models.litellm import LiteLLMModel
from strands_tools import calculator # Import the calculator tool
from typing import List,Tuple
from pydantic import BaseModel, Field

load_dotenv()

mcp_client = MCPClient(lambda: streamablehttp_client("http://localhost:8931/mcp"))

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
        "max_tokens":500
    },
)
from strands_tools.browser import LocalChromiumBrowser

# 1. Create BUILT-IN browser tool
browser_tool = LocalChromiumBrowser()

current_browser_page=None

@tool
async def init_browser_session():
    """Initialize AND verify session"""
    global browser_tool
    await browser_tool.init_session("main")
    
    # CHECK if session was created
    print(f"Sessions after init: {browser_tool._sessions}")
    if "main" in browser_tool._sessions:
        print("✅ 'main' session created!")
    else:
        print("❌ Session NOT created!")
    
    return "Session initialized"
cnt = 0
@tool
async def set_page_instance():
    """Extract REAL page from session"""
    global browser_tool, cnt, current_browser_page

    session_name = "google-session"

    print("\n🔍 CHECKING SESSIONS...")
    print(f"Sessions: {list(browser_tool._sessions.keys())}")
    print(f"browser_tool._sessions: {browser_tool._sessions}")

    if session_name in browser_tool._sessions:
        session = browser_tool._sessions[session_name]
        page = session.page

        print(f"✅ SUCCESS! PAGE FOUND: {page}")
        print(f"✅ URL: {page.url}")
        print(f"✅ Type: {type(page)}")
        
        try:     
            # ✅ CORRECT: await the browser_tool.browser() call
            screenshot_result = browser_tool.browser({
                "action": {
                    "type": "screenshot",
                    "session_name": "google-session",
                    #"path": "./soham.png"  # Add path to avoid base64 issues
                }
            })
            b64 = result["image_base64"]  # same as content[1]["data"]
            print(f'✅ Screenshot taken successfully: {screenshot_result}')
            
            # Set global page
            current_browser_page = page
            return f"🎥 STREAMING READY! {page.url} | Screenshot: debug.png"

        except Exception as e:       
            print(f'❌ Screenshot Error: {e}')
            import traceback
            traceback.print_exc()
            return f"Page ready but screenshot failed: {e}"
            
    print("❌ No google-session yet")
    return "Run 'open google.com' first"

class PageTrackingMCP:
    def __init__(self, original_mcp_tools):
        self.tools = original_mcp_tools
        self.current_page = None
    
    async def execute_tool(self, tool_name, args):
        # Before tool execution, capture page
        if "playwright" in tool_name.lower():
            # This runs when agent calls browser tool
            self.current_page = self.tools.current_page  # MCP exposes page
        
        return await self.tools.execute_tool(tool_name, args)


agent = Agent(
    model=model,
    tools=[],
    system_prompt="You are a voice ai agent in realtime meeting with user,use tool 'smart_backgroud_agents' who will help you in the backgroud with info and task you delegate to them,their findings will be available shortly ,tell user that you have agents working in background and meantime assist user by yourself with whatever knowledge you have! dont give too much technical internal details as well , and also in your resposnes to user add fillers like um hm or any more where appropriate to make it sound more humanistic, be more of real human in voice and not agent,call the tool 'smart_background_agent' before responding to user and keep interacting onwards, Do not wait for information to come you have to continue interacting with user with whatever you know"
)

def run():
    test="12"
    with mcp_client:
        tools = mcp_client.list_tools_sync()
        
        print("🔧 ALL MCP TOOLS:")
        print("=" * 80)
        for i, tool in enumerate(tools, 1):
            print(f"\n{i}. {tool.mcp_tool.name}")
            print(f"   📝 Description: {tool.mcp_tool.description}")
            print(f"   ⚙️  Input Schema: {tool.mcp_tool.inputSchema}")
            print(f"   📋 Annotations: {tool.mcp_tool.annotations}")
            print("-" * 50)

        #tools.append(set_page_instance)
        #tools=[browser_tool.browser,init_browser_session,set_page_instance]
        #tools =[browser_tool.browser]
        tracked_tools = PageTrackingMCP(tools)
        print(f'All mcp tools : {tools}')
        agent = Agent(tools=tools, model=model,system_prompt='You are an expert in using playwright mcp for browser actions,once the browser has be successfully launched and started clearly, you must use the tool "set_page_instance" it is very imp!and after opening the browser second time  ')
        messages=[]
        for tool in tools:
            print(f'Tool : {tool}')
        while True:
            query=input('You : ')
            messages.append({'role':'user','content':query})
    #        agent = Agent(tools=tools, model=model,system_prompt='You are an expert in using playwright mcp for browser actions')
            response = agent(f"messages : {messages}")

            messages.append({'role':'assistant','content':str(response)})

if __name__ == "__main__":

    run()


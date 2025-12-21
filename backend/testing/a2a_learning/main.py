from strands import Agent
from strands.multiagent.a2a import A2AServer
import uvicorn

# Create your agent and A2A server
agent = Agent(name="My Agent", description="A customizable agent", callback_handler=None)
a2a_server = A2AServer(agent=agent)
a2a_server.serve()

# Access the underlying FastAPI app
# fastapi_app = a2a_server.to_fastapi_app()
# Add custom middleware, routes, or configuration
# fastapi_app.add_middleware(...)

# Or access the Starlette app
# starlette_app = a2a_server.to_starlette_app()
# Customize as needed

# You can then serve the customized app directly

# uvicorn.run(a2a_server, host="127.0.0.1", port=9000)
# if __name__ == "__main__":

# LangGraph Platform Runner
#
# Executes agents deployed to LangGraph Platform.
# Supports threads (conversation persistence) and streaming.
#
# Usage:
#     runner = LangGraphRunner(deployment_url="https://...", assistant_id="agent")
#     result = runner.run({"messages": [{"role": "user", "content": "Hello"}]})

import os
import asyncio
from typing import Any, Dict, List, Optional, Union

from src.deployment.strategies.base import Runner


class LangGraphRunner(Runner):
    """
    Runs agents deployed to LangGraph Platform.
    
    Features:
    - Thread management (conversation persistence)
    - Streaming responses
    - Checkpointing support
    
    Usage:
        runner = LangGraphRunner(
            deployment_url="https://your-deployment.langchain.app",
            assistant_id="agent"
        )
        result = runner.run({"messages": [{"role": "user", "content": "Hello!"}]})
    """
    
    def __init__(
        self,
        deployment_url: str = None,
        assistant_id: str = "agent",
        code_path: str = None,
        **kwargs,  # Accept extra params from run_interface
    ):
        """
        Initialize the LangGraph runner.
        
        Args:
            deployment_url: URL of the deployed LangGraph agent
            assistant_id: ID of the assistant/graph to call
            code_path: Path to the code directory
            **kwargs: Additional parameters (ignored)
        """
        self.deployment_url = deployment_url or os.environ.get("LANGGRAPH_API_URL")
        self.assistant_id = assistant_id
        self.code_path = code_path
        self._api_key = os.environ.get("LANGSMITH_API_KEY")
        self._client = None
        self._current_thread_id = None
        self._deployed = False
        self._logs: List[str] = []
        
        self._connect()
    
    def _connect(self) -> None:
        """Initialize connection to LangGraph Platform."""
        if not self._api_key:
            self._logs.append("LANGSMITH_API_KEY not set")
            return
        
        if not self.deployment_url:
            self._logs.append("Deployment URL not set")
            self._logs.append(f"To deploy: cd {self.code_path} && langgraph deploy")
            return
        
        try:
            from langgraph_sdk import get_client
            self._client = get_client(url=self.deployment_url)
            self._deployed = True
            self._logs.append(f"Connected to LangGraph Platform: {self.deployment_url}")
        except ImportError:
            self._logs.append("langgraph-sdk not installed. Run: pip install langgraph-sdk")
        except Exception as e:
            self._logs.append(f"Connection error: {e}")
    
    def _ensure_thread_sync(self) -> str:
        """Create or reuse a thread."""
        if self._current_thread_id:
            return self._current_thread_id
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._create_thread())
                    self._current_thread_id = future.result()
            else:
                self._current_thread_id = loop.run_until_complete(self._create_thread())
        except RuntimeError:
            self._current_thread_id = asyncio.run(self._create_thread())
        
        return self._current_thread_id
    
    async def _create_thread(self) -> str:
        """Create a new thread."""
        thread = await self._client.threads.create()
        thread_id = thread.get("thread_id") or thread.get("id")
        self._logs.append(f"Created thread: {thread_id}")
        return thread_id
    
    def run(self, inputs: Union[Dict, str, bytes]) -> Any:
        """
        Call the LangGraph agent.
        
        Args:
            inputs: Input data
            
        Returns:
            Agent response
        """
        if not self._deployed or not self._client:
            return {
                "status": "error",
                "error": "LangGraph Platform not connected",
                "instructions": [
                    "1. Set LANGSMITH_API_KEY",
                    "2. Deploy: langgraph deploy",
                    "3. Set LANGGRAPH_API_URL",
                ],
                "deploy_command": f"cd {self.code_path} && langgraph deploy",
            }
        
        messages = self._prepare_messages(inputs)
        
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self._run_async(messages))
                        return future.result()
                else:
                    return loop.run_until_complete(self._run_async(messages))
            except RuntimeError:
                return asyncio.run(self._run_async(messages))
        except Exception as e:
            # Handle common LangGraph SDK errors with helpful messages
            error_type = type(e).__name__
            error_msg = str(e) or error_type
            
            # NotFoundError means the agent isn't deployed
            if error_type == "NotFoundError":
                error_msg = (
                    f"Agent '{self.assistant_id}' not found at {self.deployment_url}. "
                    f"The agent may not be deployed yet. "
                    f"Deploy with: cd {self.code_path} && langgraph deploy"
                )
            
            self._logs.append(f"Run error: {error_type}: {error_msg}")
            return {"status": "error", "error": error_msg}
    
    def _prepare_messages(self, inputs: Union[Dict, str, bytes]) -> List[Dict]:
        """Convert inputs to message format."""
        if isinstance(inputs, str):
            return [{"role": "user", "content": inputs}]
        elif isinstance(inputs, bytes):
            return [{"role": "user", "content": inputs.decode("utf-8")}]
        elif isinstance(inputs, dict):
            if "messages" in inputs:
                return inputs["messages"]
            elif "content" in inputs:
                return [{"role": "user", "content": inputs["content"]}]
            elif "text" in inputs:
                return [{"role": "user", "content": inputs["text"]}]
            else:
                import json
                return [{"role": "user", "content": json.dumps(inputs)}]
        return [{"role": "user", "content": str(inputs)}]
    
    async def _run_async(self, messages: List[Dict]) -> Dict[str, Any]:
        """Async run implementation."""
        thread_id = self._ensure_thread_sync()
        
        self._logs.append(f"Running agent on thread {thread_id}")
        
        result = await self._client.runs.wait(
            thread_id,
            self.assistant_id,
            input={"messages": messages},
        )
        
        self._logs.append("Agent run completed")
        
        if isinstance(result, dict):
            return {"status": "success", "output": result}
        return {"status": "success", "output": {"result": result}}
    
    def new_thread(self) -> None:
        """Start a new conversation thread."""
        self._current_thread_id = None
        self._logs.append("Thread reset")
    
    def stop(self) -> None:
        """Cleanup resources."""
        self._client = None
        self._current_thread_id = None
        self._logs.append("Disconnected")
    
    def is_healthy(self) -> bool:
        """Check if connected."""
        return self._deployed and self._client is not None
    
    def get_logs(self) -> str:
        """Get runner logs."""
        return "\n".join(self._logs)
    
    def get_deploy_command(self) -> str:
        """Get deploy command."""
        if self.code_path:
            return f"cd {self.code_path} && langgraph deploy"
        return "langgraph deploy"


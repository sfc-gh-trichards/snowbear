"""
Snowflake Cortex Agent Client
Handles communication with your Snowflake data agent
"""

import os
import uuid
import httpx
from typing import Optional


class SnowflakeAgent:
    def __init__(self):
        self.account_url = os.getenv("SNOWFLAKE_ACCOUNT_URL")
        self.pat = os.getenv("SNOWFLAKE_PAT")
        self.database = os.getenv("SNOWFLAKE_AGENT_DATABASE", "SNOWFLAKE_INTELLIGENCE")
        self.schema = os.getenv("SNOWFLAKE_AGENT_SCHEMA", "AGENTS")
        self.agent_name = os.getenv("SNOWFLAKE_AGENT_NAME", "DONUT_STORE_AGENT")
        
        if not self.account_url or not self.pat:
            raise ValueError("Missing SNOWFLAKE_ACCOUNT_URL or SNOWFLAKE_PAT in environment")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.pat}",
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        }

    async def create_thread(self, client: httpx.AsyncClient) -> str:
        """Create a new conversation thread"""
        try:
            resp = await client.post(
                f"{self.account_url}/api/v2/cortex/threads",
                headers=self.headers,
                json={"origin_application": "SnowbearPi"},
            )
            resp.raise_for_status()
            data = resp.json()
            # Handle different response formats
            if "id" in data:
                return data["id"]
            elif "thread_id" in data:
                return data["thread_id"]
            elif isinstance(data, dict) and len(data) == 1:
                return list(data.values())[0]
            else:
                print(f"   ⚠️ Thread response: {data}")
                # Fall back to UUID
                return str(uuid.uuid4())
        except Exception as e:
            print(f"   ⚠️ Thread creation failed ({e}), using UUID")
            return str(uuid.uuid4())

    async def query(self, user_text: str) -> str:
        """
        Send a query to the Snowflake Cortex Agent and get the response.
        This may take 15-30 seconds depending on query complexity.
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Create thread
            thread_id = await self.create_thread(client)
            
            # Build request
            path = (
                f"/api/v2/databases/{self.database}"
                f"/schemas/{self.schema}"
                f"/agents/{self.agent_name}:run"
            )
            
            body = {
                "thread_id": thread_id,
                "parent_message_id": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_text}],
                    }
                ],
                "tool_choice": {"type": "auto"},
            }
            
            # Make request (SSE stream)
            resp = await client.post(
                f"{self.account_url}{path}",
                headers={**self.headers, "Accept": "text/event-stream"},
                json=body,
            )
            resp.raise_for_status()
            
            # Parse SSE response
            sse_text = resp.text
            final_response = None
            last_event = None
            all_events = []  # For debugging
            
            for line in sse_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("event:"):
                    last_event = line[6:].strip()
                    all_events.append(last_event)
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        import json
                        try:
                            parsed = json.loads(data_str)
                            # Look for text content in any event
                            if last_event == "response":
                                final_response = parsed
                                break
                            # Also check for content in other events
                            if isinstance(parsed, dict) and "content" in parsed:
                                final_response = parsed
                        except json.JSONDecodeError:
                            pass
            
            if not final_response:
                print(f"   ⚠️ Events received: {all_events}")
                print(f"   ⚠️ Raw response (first 500 chars): {sse_text[:500]}")
                raise ValueError("No response from Snowflake agent")
            
            # Extract text content
            text_part = None
            for content in final_response.get("content", []):
                if content.get("type") == "text":
                    text_part = content.get("text")
                    break
            
            return text_part or str(final_response)


# Singleton instance
_agent: Optional[SnowflakeAgent] = None


def get_agent() -> SnowflakeAgent:
    global _agent
    if _agent is None:
        _agent = SnowflakeAgent()
    return _agent


async def query_snowflake(question: str) -> str:
    """Convenience function to query Snowflake"""
    agent = get_agent()
    return await agent.query(question)


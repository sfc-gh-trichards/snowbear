"""
Snowflake Cortex Agent Client
Handles communication with your Snowflake data agent

API Reference: POST /api/v2/databases/{database}/schemas/{schema}/agents/{name}:run
"""

import json
import os
import httpx
from typing import Optional


class SnowflakeAgent:
    def __init__(self):
        self.account_url = os.getenv("SNOWFLAKE_ACCOUNT_URL", "").rstrip("/")
        self.pat = os.getenv("SNOWFLAKE_PAT")
        self.database = os.getenv("SNOWFLAKE_AGENT_DATABASE")
        self.schema = os.getenv("SNOWFLAKE_AGENT_SCHEMA")
        self.agent_name = os.getenv("SNOWFLAKE_AGENT_NAME")
        
        # Validate required fields
        missing = []
        if not self.account_url:
            missing.append("SNOWFLAKE_ACCOUNT_URL")
        if not self.pat:
            missing.append("SNOWFLAKE_PAT")
        if not self.database:
            missing.append("SNOWFLAKE_AGENT_DATABASE")
        if not self.schema:
            missing.append("SNOWFLAKE_AGENT_SCHEMA")
        if not self.agent_name:
            missing.append("SNOWFLAKE_AGENT_NAME")
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.pat}",
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        }
        
        print(f"   ✓ Snowflake Agent configured:")
        print(f"     Account: {self.account_url}")
        print(f"     Agent: {self.database}.{self.schema}.{self.agent_name}")

    async def query(self, user_text: str) -> str:
        """
        Send a query to the Snowflake Cortex Agent and get the response.
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Build the agent run endpoint
            url = (
                f"{self.account_url}/api/v2/databases/{self.database}"
                f"/schemas/{self.schema}/agents/{self.agent_name}:run"
            )
            
            # Request body - try without thread_id first (simpler)
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_text}],
                    }
                ],
                "tool_choice": {"type": "auto"},
            }
            
            print(f"   → POST {url}")
            print(f"   → Body: {json.dumps(body, indent=2)[:200]}...")
            
            try:
                resp = await client.post(
                    url,
                    headers={**self.headers, "Accept": "text/event-stream"},
                    json=body,
                )
            except Exception as e:
                print(f"   ✗ Request failed: {e}")
                raise
            
            print(f"   ← Status: {resp.status_code}")
            
            if resp.status_code != 200:
                error_text = resp.text[:500]
                print(f"   ✗ Error response: {error_text}")
                raise ValueError(f"Snowflake API error {resp.status_code}: {error_text}")
            
            # Parse SSE response
            response_text = resp.text
            print(f"   ← Response length: {len(response_text)} chars")
            
            # Debug: show first part of response
            if len(response_text) < 1000:
                print(f"   ← Full response: {response_text}")
            else:
                print(f"   ← Response preview: {response_text[:500]}...")
            
            return self._parse_sse_response(response_text)
    
    def _parse_sse_response(self, sse_text: str) -> str:
        """Parse Server-Sent Events response to extract the final answer."""
        final_text = None
        current_event = None
        
        for line in sse_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()
                if not data_str:
                    continue
                
                try:
                    data = json.loads(data_str)
                    
                    # Look for text content in the response
                    if isinstance(data, dict):
                        # Check for content array
                        content = data.get("content", [])
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                if text:
                                    final_text = text
                        
                        # Also check for direct text field
                        if "text" in data and data["text"]:
                            final_text = data["text"]
                        
                        # Check for message content
                        if "message" in data:
                            msg = data["message"]
                            if isinstance(msg, dict) and "content" in msg:
                                for item in msg.get("content", []):
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        final_text = item.get("text", final_text)
                
                except json.JSONDecodeError as e:
                    print(f"   ⚠ JSON parse error: {e} for data: {data_str[:100]}")
                    continue
        
        if not final_text:
            # If we couldn't parse it, return the raw response for debugging
            raise ValueError(f"Could not extract text from response. Raw: {sse_text[:500]}")
        
        return final_text


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

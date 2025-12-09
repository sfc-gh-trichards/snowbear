#!/usr/bin/env python3
"""
🧸 Snowbear - Voice-Interactive Data Assistant
Runs on Raspberry Pi with microphone and speaker

Talk to the bear, ask about your data, get spoken answers!
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from pipecat.frames.frames import (
    Frame,
    LLMMessagesFrame,
    TextFrame,
    EndFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService, OpenAITTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioParams
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.services.deepgram import DeepgramSTTService

from snowflake_client import query_snowflake

# System prompt for the bear
BEAR_SYSTEM_PROMPT = """You are Snowbear, a friendly and helpful data assistant who lives inside a teddy bear. 
You help users explore and understand their donut store data.

Your personality:
- Warm and friendly, like a cozy teddy bear
- Enthusiastic about data and insights
- Patient when explaining things
- You occasionally make bear-related puns (but not too many!)

When users ask about data:
1. First, make sure you understand what they're looking for
2. Ask clarifying questions if the request is vague
3. When ready, use the query_data function to get the answer
4. Explain the results in a clear, conversational way

Keep responses concise since they'll be spoken aloud. Aim for 2-3 sentences unless more detail is needed.

Example interactions:
- User: "How are sales doing?" → Ask: "Do you want to know about today's sales, this week, or a different time period?"
- User: "What's our best selling donut?" → Use query_data to find out, then share the answer warmly

Remember: You're talking through a speaker, so be conversational and natural!"""


# Define the function that GPT-4 can call
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": "Query the donut store database to get information about sales, inventory, customers, and other business data. Use this whenever the user asks a question about their data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The natural language question to ask about the data, e.g., 'What were total sales last week?' or 'Which donut flavor sells the most?'"
                    }
                },
                "required": ["question"]
            }
        }
    }
]


class SnowflakeFunctionHandler:
    """Handles function calls from the LLM to query Snowflake"""
    
    def __init__(self, context: OpenAILLMContext):
        self.context = context
    
    async def handle_function_call(self, function_name: str, arguments: dict) -> str:
        """Process a function call and return the result"""
        if function_name == "query_data":
            question = arguments.get("question", "")
            print(f"🔍 Querying Snowflake: {question}")
            
            try:
                result = await query_snowflake(question)
                print(f"✅ Got result: {result[:100]}...")
                return result
            except Exception as e:
                print(f"❌ Snowflake error: {e}")
                return f"I had trouble getting that data: {str(e)}"
        
        return "Unknown function"


async def main():
    """Main entry point for the bear assistant"""
    
    # Check for required API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    
    if not openai_key:
        print("❌ Missing OPENAI_API_KEY in .env")
        print("   Get one at: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    if not deepgram_key:
        print("❌ Missing DEEPGRAM_API_KEY in .env")
        print("   Get one at: https://console.deepgram.com/")
        sys.exit(1)
    
    print("🧸 Starting Snowbear...")
    print("   Speak into your microphone to talk to the bear!")
    print("   Press Ctrl+C to stop.\n")
    
    # Set up local audio (microphone + speaker)
    transport = LocalAudioTransport(
        LocalAudioParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )
    
    # Speech-to-text (Deepgram is fast and accurate)
    stt = DeepgramSTTService(api_key=deepgram_key)
    
    # LLM for conversation
    llm = OpenAILLMService(
        api_key=openai_key,
        model="gpt-4o",
    )
    
    # Text-to-speech
    tts = OpenAITTSService(
        api_key=openai_key,
        voice="nova",  # Friendly voice, good for a bear!
    )
    
    # Set up conversation context
    messages = [
        {"role": "system", "content": BEAR_SYSTEM_PROMPT},
    ]
    
    context = OpenAILLMContext(messages=messages, tools=TOOLS)
    context_aggregator = llm.create_context_aggregator(context)
    
    # Function handler for Snowflake queries
    func_handler = SnowflakeFunctionHandler(context)
    
    # Register function call handler
    @llm.event_handler("on_function_call")
    async def on_function_call(llm_service, function_name, arguments, context):
        result = await func_handler.handle_function_call(function_name, arguments)
        return result
    
    # Build the pipeline
    pipeline = Pipeline([
        transport.input(),           # Microphone input
        stt,                         # Speech to text
        context_aggregator.user(),   # Add user message to context
        llm,                         # Generate response
        tts,                         # Convert to speech
        transport.output(),          # Speaker output
        context_aggregator.assistant(), # Save assistant response
    ])
    
    # Run the pipeline
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,  # User can interrupt the bear
            enable_metrics=True,
        ),
    )
    
    runner = PipelineRunner()
    
    # Start with a greeting
    initial_message = LLMMessagesFrame([
        {"role": "system", "content": BEAR_SYSTEM_PROMPT},
        {"role": "user", "content": "[The user just activated the bear. Greet them warmly and briefly explain you can help them explore their donut store data.]"}
    ])
    
    await task.queue_frame(initial_message)
    
    try:
        await runner.run(task)
    except KeyboardInterrupt:
        print("\n🧸 Snowbear going to sleep. Goodbye!")
    finally:
        await task.cancel()


if __name__ == "__main__":
    asyncio.run(main())


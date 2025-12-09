#!/usr/bin/env python3
"""
🧸 Snowbear - Voice-Interactive Data Assistant
Runs on Raspberry Pi with microphone and speaker

Talk to the bear, ask about your data, get spoken answers!
"""

import asyncio
import io
import json
import os
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from openai import OpenAI
from dotenv import load_dotenv

from snowflake_client import query_snowflake

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Audio settings
SAMPLE_RATE = 16000  # Whisper likes 16kHz
CHANNELS = 1
SILENCE_THRESHOLD = 0.01  # Adjust based on your mic
SILENCE_DURATION = 1.5  # Seconds of silence to stop recording
MAX_RECORDING_DURATION = 30  # Max seconds to record

# System prompt for the bear
SYSTEM_PROMPT = """You are Snowbear, a friendly and helpful data assistant who lives inside a teddy bear. 
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

Remember: You're talking through a speaker, so be conversational and natural!"""

# Function definition for GPT-4
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
                        "description": "The natural language question to ask about the data"
                    }
                },
                "required": ["question"]
            }
        }
    }
]

# Conversation history
conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]


def record_audio() -> np.ndarray:
    """Record audio from microphone until silence is detected."""
    print("🎤 Listening... (speak now)")
    
    audio_chunks = []
    silent_chunks = 0
    chunks_for_silence = int(SILENCE_DURATION * SAMPLE_RATE / 1024)
    max_chunks = int(MAX_RECORDING_DURATION * SAMPLE_RATE / 1024)
    
    def callback(indata, frames, time, status):
        nonlocal silent_chunks
        if status:
            print(f"Audio status: {status}")
        
        volume = np.abs(indata).mean()
        if volume < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0
        
        audio_chunks.append(indata.copy())
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                        dtype='float32', blocksize=1024, callback=callback):
        while silent_chunks < chunks_for_silence and len(audio_chunks) < max_chunks:
            sd.sleep(100)
    
    if len(audio_chunks) < 5:  # Too short
        print("   (no speech detected)")
        return np.array([])
    
    print("   ✓ Got it!")
    return np.concatenate(audio_chunks)


def transcribe_audio(audio: np.ndarray) -> str:
    """Convert audio to text using OpenAI Whisper."""
    if len(audio) == 0:
        return ""
    
    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(temp_path, SAMPLE_RATE, audio_int16)
    
    try:
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcript.strip()
    finally:
        os.unlink(temp_path)


async def handle_function_call(function_name: str, arguments: dict) -> str:
    """Execute a function call from the LLM."""
    if function_name == "query_data":
        question = arguments.get("question", "")
        print(f"   🔍 Querying Snowflake: {question}")
        
        try:
            result = await query_snowflake(question)
            print(f"   ✅ Got data!")
            return result
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return f"Error querying data: {str(e)}"
    
    return "Unknown function"


async def get_response(user_text: str) -> str:
    """Get a response from GPT-4, handling any function calls."""
    conversation_history.append({"role": "user", "content": user_text})
    
    # First API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history,
        tools=TOOLS,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    
    # Handle function calls
    while message.tool_calls:
        conversation_history.append(message)
        
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            result = await handle_function_call(function_name, arguments)
            
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
        
        # Get next response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            tools=TOOLS,
            tool_choice="auto"
        )
        message = response.choices[0].message
    
    assistant_text = message.content
    conversation_history.append({"role": "assistant", "content": assistant_text})
    
    return assistant_text


def speak(text: str):
    """Convert text to speech and play it."""
    print(f"🔊 Speaking: {text[:80]}...")
    
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",  # Friendly voice for the bear
        input=text
    )
    
    # Save to temp file and play
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name
        response.stream_to_file(temp_path)
    
    try:
        # Use ffmpeg to convert and play (most reliable on Pi)
        os.system(f"ffplay -nodisp -autoexit -loglevel quiet {temp_path}")
    finally:
        os.unlink(temp_path)


async def main():
    """Main conversation loop."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY in .env file")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("🧸 SNOWBEAR - Voice Data Assistant")
    print("="*50)
    print("\nSpeak to ask questions about your donut store data!")
    print("Press Ctrl+C to exit.\n")
    
    # Initial greeting
    greeting = "Hi there! I'm Snowbear, your friendly data assistant. Ask me anything about your donut store, and I'll dig into the data for you!"
    speak(greeting)
    
    while True:
        try:
            # Record user speech
            audio = record_audio()
            
            if len(audio) == 0:
                continue
            
            # Transcribe
            print("   🔄 Transcribing...")
            user_text = transcribe_audio(audio)
            
            if not user_text:
                continue
            
            print(f"   📝 You said: \"{user_text}\"")
            
            # Get response
            print("   🤔 Thinking...")
            response_text = await get_response(user_text)
            
            # Speak response
            speak(response_text)
            print()
            
        except KeyboardInterrupt:
            print("\n\n🧸 Goodbye! Sweet dreams!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            speak("Oops, I had a little hiccup. Could you try again?")


if __name__ == "__main__":
    asyncio.run(main())

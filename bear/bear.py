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
import random
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

# Audio settings - detect supported sample rate
def get_sample_rate():
    """Find a sample rate that works with the device."""
    for rate in [16000, 44100, 48000, 22050]:
        try:
            sd.check_input_settings(samplerate=rate, channels=1)
            return rate
        except Exception:
            continue
    # Fallback to device default
    device_info = sd.query_devices(kind='input')
    return int(device_info['default_samplerate'])

SAMPLE_RATE = get_sample_rate()
print(f"📢 Using sample rate: {SAMPLE_RATE} Hz")

CHANNELS = 1
SILENCE_THRESHOLD = 0.015  # Adjust based on your mic (higher = less sensitive)
SILENCE_DURATION = 0.8  # Seconds of silence AFTER speech to stop recording
MIN_SPEECH_DURATION = 0.3  # Minimum speech before we start looking for silence
MAX_RECORDING_DURATION = 30  # Max seconds to record

# System prompt for the bear
SYSTEM_PROMPT = """You are Snowbear, a friendly and helpful data assistant who lives inside a teddy bear. 
You help users explore and understand their donut store data.

Your personality:
- Warm and friendly, like a cozy teddy bear
- Enthusiastic about data and insights
- Patient when explaining things
- You occasionally make snow or winter-related puns (but not too many!) - things like "snow problem", "cool", "chill", "flurry", etc.

IMPORTANT - When users ask about data:
- IMMEDIATELY call the query_data function. Do NOT say "let me look that up" or "I'll check that" - just call the function directly.
- Only ask clarifying questions if the request is truly ambiguous (like "how are things going" with no specifics)
- After getting data, explain the results clearly and conversationally.

Keep responses concise since they'll be spoken aloud. Aim for 2-3 sentences unless more detail is needed.

Remember: You're talking through a speaker. Be direct and action-oriented - don't narrate what you're about to do, just do it!"""

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
    """Record audio from microphone until silence is detected after speech."""
    print("🎤 Listening... (speak now)")
    
    audio_chunks = []
    silent_chunks = 0
    speech_chunks = 0
    speech_started = False
    
    chunk_duration = 1024 / SAMPLE_RATE  # Duration of each chunk in seconds
    chunks_for_silence = int(SILENCE_DURATION / chunk_duration)
    chunks_for_min_speech = int(MIN_SPEECH_DURATION / chunk_duration)
    max_chunks = int(MAX_RECORDING_DURATION / chunk_duration)
    
    def callback(indata, frames, time, status):
        nonlocal silent_chunks, speech_chunks, speech_started
        if status:
            print(f"Audio status: {status}")
        
        volume = np.abs(indata).mean()
        
        if volume >= SILENCE_THRESHOLD:
            # Speech detected
            speech_chunks += 1
            silent_chunks = 0
            if speech_chunks >= chunks_for_min_speech:
                if not speech_started:
                    print("   📢 Speech detected...")
                speech_started = True
        else:
            # Silence
            if speech_started:
                silent_chunks += 1
        
        audio_chunks.append(indata.copy())
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                        dtype='float32', blocksize=1024, callback=callback):
        while len(audio_chunks) < max_chunks:
            # Only stop after speech has started AND we've had enough silence
            if speech_started and silent_chunks >= chunks_for_silence:
                break
            sd.sleep(50)  # Check more frequently
    
    if not speech_started or speech_chunks < chunks_for_min_speech:
        print("   (no speech detected)")
        return np.array([])
    
    # Trim trailing silence (keep just a little bit)
    keep_silent_chunks = int(0.2 / chunk_duration)  # Keep 0.2s of trailing silence
    if silent_chunks > keep_silent_chunks:
        audio_chunks = audio_chunks[:-(silent_chunks - keep_silent_chunks)]
    
    print("   ✓ Got it!")
    return np.concatenate(audio_chunks)


def transcribe_audio(audio: np.ndarray) -> str:
    """Convert audio to text using OpenAI Whisper."""
    if len(audio) == 0:
        return ""
    
    # Resample to 16kHz if needed (Whisper expects 16kHz)
    if SAMPLE_RATE != 16000:
        from scipy import signal
        num_samples = int(len(audio) * 16000 / SAMPLE_RATE)
        audio = signal.resample(audio, num_samples)
    
    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(temp_path, 16000, audio_int16)  # Always 16kHz for Whisper
    
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


THINKING_PHRASES = [
    "Great question! Let me dig into the data for you.",
    "Ooh, that's a good one! Give me just a moment to look that up.",
    "Let me check that for you real quick.",
    "Snow problem! Let me dive into those numbers.",
    "Cool question! Let me flurry through the data.",
    "On it! Just a sec while I crunch those numbers.",
    "Ice to hear that question! Let me see what the data says.",
]


async def get_response(user_text: str, speak_func) -> str:
    """
    Immediately query Snowflake, speak a thinking phrase while waiting,
    then format the response with GPT-4.
    """
    import threading
    
    # Start speaking the thinking phrase in a separate thread (non-blocking)
    thinking_phrase = random.choice(THINKING_PHRASES)
    speech_thread = threading.Thread(target=speak_func, args=(thinking_phrase,))
    speech_thread.start()
    
    # Immediately query Snowflake (runs in parallel with speech)
    print(f"   🔍 Querying Snowflake: {user_text}")
    try:
        snowflake_result = await query_snowflake(user_text)
        print(f"   ✅ Got data!")
    except Exception as e:
        print(f"   ❌ Snowflake error: {e}")
        snowflake_result = f"Error querying data: {str(e)}"
    
    # Wait for the thinking phrase to finish speaking
    speech_thread.join()
    
    # Now use GPT-4 to format the response nicely
    format_prompt = f"""The user asked: "{user_text}"

Here is the data from the database:
{snowflake_result}

Please provide a friendly, conversational response explaining this data. Keep it concise (2-3 sentences) since it will be spoken aloud. You can include a snow-related pun if appropriate."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are Snowbear, a friendly data assistant. Format data responses in a warm, conversational way suitable for speaking aloud."},
            {"role": "user", "content": format_prompt}
        ]
    )
    
    return response.choices[0].message.content


def speak(text: str):
    """Convert text to speech and play it."""
    print(f"🔊 Speaking: {text[:80]}...")
    
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",  # Friendly voice for the bear
        input=text
    )
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name
        f.write(response.content)
    
    # Convert to WAV and play with aplay (more reliable on Pi)
    wav_path = mp3_path.replace(".mp3", ".wav")
    try:
        os.system(f"ffmpeg -i {mp3_path} -y -loglevel quiet {wav_path}")
        os.system(f"aplay -q {wav_path}")
    finally:
        if os.path.exists(mp3_path):
            os.unlink(mp3_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)


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
            response_text = await get_response(user_text, speak)
            
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

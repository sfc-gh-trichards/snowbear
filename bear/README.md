# 🧸 Snowbear - Voice-Interactive Data Assistant

Talk to a teddy bear, ask about your donut store data, get spoken answers!

## Quick Setup on Raspberry Pi

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y git ffmpeg libportaudio2
```

### 2. Install uv (fast Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 3. Clone the repo

```bash
cd ~
git clone https://github.com/sfc-gh-trichards/snowbear.git
cd snowbear/bear
```

### 4. Set up Python environment

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 5. Create your `.env` file

```bash
nano ~/snowbear/.env
```

Add your keys:
```
OPENAI_API_KEY=sk-...
SNOWFLAKE_ACCOUNT_URL=https://your-account.snowflakecomputing.com
SNOWFLAKE_PAT=your-token
SNOWFLAKE_AGENT_DATABASE=SNOWFLAKE_INTELLIGENCE
SNOWFLAKE_AGENT_SCHEMA=AGENTS
SNOWFLAKE_AGENT_NAME=DONUT_STORE_AGENT
```

Save: `Ctrl+O`, Enter, `Ctrl+X`

### 6. Run the bear!

```bash
cd ~/snowbear/bear
source .venv/bin/activate
python bear.py
```

## APIs Needed

| API | What it does | Cost |
|-----|--------------|------|
| **OpenAI** | Whisper (STT) + GPT-4o + TTS | ~$5-10 for hackathon |
| **Snowflake** | Your data queries | You have this |

Get OpenAI key: https://platform.openai.com/api-keys

## How It Works

```
[Microphone] → [Whisper STT] → [GPT-4o] → [Snowflake] → [GPT-4o] → [TTS] → [Speaker]
```

1. You speak into the microphone
2. OpenAI Whisper transcribes your speech
3. GPT-4o understands your question and asks clarifying questions if needed
4. When ready, it queries your Snowflake Cortex Agent
5. GPT-4o formats the response conversationally
6. OpenAI TTS speaks the answer through your speaker

## Troubleshooting

### No audio input detected
```bash
# List audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Test recording
arecord -d 3 test.wav && aplay test.wav
```

### ffplay not found
```bash
sudo apt install ffmpeg
```

### Adjust microphone sensitivity
Edit `bear.py` and change `SILENCE_THRESHOLD` (default 0.01). Higher = less sensitive.

# 🧸 Snowbear - Voice-Interactive Data Assistant

Talk to a teddy bear, ask about your donut store data, get spoken answers!

## Hardware Requirements

- Raspberry Pi (3B+ or newer recommended)
- USB Microphone or USB sound card with mic
- Speaker (3.5mm jack or USB)
- Your teddy bear to put it all in! 🧸

## Quick Setup

### 1. SSH into your Raspberry Pi

```bash
ssh your-username@your-pi-address
```

### 2. Install system dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install audio libraries
sudo apt install -y python3-pip python3-venv portaudio19-dev python3-pyaudio

# Test your audio devices
arecord -l  # List recording devices
aplay -l    # List playback devices
```

### 3. Clone and set up the project

```bash
# Navigate to project
cd /path/to/snowbear/bear

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure API Keys

Edit the `.env` file in the parent directory (`snowbear/.env`):

```env
# OpenAI - for GPT-4 and TTS
OPENAI_API_KEY=sk-...

# Deepgram - for fast speech-to-text
DEEPGRAM_API_KEY=...

# Snowflake - your existing config
SNOWFLAKE_ACCOUNT_URL=https://your-account.snowflakecomputing.com
SNOWFLAKE_PAT=your-programmatic-access-token
SNOWFLAKE_AGENT_DATABASE=SNOWFLAKE_INTELLIGENCE
SNOWFLAKE_AGENT_SCHEMA=AGENTS
SNOWFLAKE_AGENT_NAME=DONUT_STORE_AGENT
```

### 5. Test your microphone and speaker

```bash
# Record 5 seconds of audio
arecord -d 5 test.wav

# Play it back
aplay test.wav
```

### 6. Run the bear!

```bash
source venv/bin/activate
python bear.py
```

## Getting API Keys

### OpenAI
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add funds to your account (~$10 is plenty for testing)

### Deepgram
1. Go to https://console.deepgram.com/
2. Sign up (free tier includes $200 credit!)
3. Create an API key

## Troubleshooting

### "No audio devices found"
```bash
# Check if your mic is detected
arecord -l

# If using USB mic, try unplugging and replugging
# Check dmesg for USB audio
dmesg | grep -i audio
```

### Audio too quiet / too loud
```bash
# Adjust mic volume
alsamixer
# Use F6 to select your sound card, then adjust levels
```

### Snowflake connection issues
Make sure your Pi can reach your Snowflake account. If you have network policies enabled in Snowflake, you may need to allowlist your Pi's IP address.

## Architecture

```
[Microphone] 
    ↓
[Silero VAD] - Detects when you're speaking
    ↓
[Deepgram STT] - Converts speech to text
    ↓
[GPT-4o] - Understands intent, asks clarifying questions
    ↓ (when ready)
[Snowflake Cortex Agent] - Queries your data
    ↓
[GPT-4o] - Formats response conversationally  
    ↓
[OpenAI TTS] - Converts to speech
    ↓
[Speaker]
```

## Tips for Hackathon Demo

1. **Start simple**: Begin with basic questions like "How many donuts did we sell today?"
2. **Prepare questions**: Have a list of interesting questions ready
3. **Good lighting**: Make sure the bear is visible!
4. **Test audio levels**: Do a sound check before demoing
5. **Have a backup**: Keep your laptop ready to type if voice fails

## Cost Estimate

For a hackathon demo (~1 hour of usage):
- OpenAI GPT-4o: ~$2-5
- OpenAI TTS: ~$1-2  
- Deepgram STT: Free tier covers it
- **Total: ~$5-10**


# LiveKit Museum Curator

This demo showcases LiveKit's powerful combination of Speech-to-Text (STT), Large Language Models (LLM), and Text-to-Speech (TTS) in a museum curator setting. It simulates a futuristic museum experience, featuring my favorite android, David from the Alien franchise. Feel free to ask him questions about the exhibits and let him make you feel like an intellectually inferior human...

## Features

- **AI Voice Agent as Museum Curator:** David is a perfect host...almost, engaging with users to provide information about museum exhibits from a time when AI was still a novelty.
- **Real-time Communication:** Utilizes LiveKit Cloud for real-time audio and soon...video interactions.
- **AI-Powered Responses:** Integrates with various AI services (Deepgram, ElevenLabs, OpenAI) for speech-to-text, text-to-speech, and intelligent conversational abilities.
- **Noise Cancellation:** Enhances audio quality for clearer interactions.
- **Turn Detection (VAD):** LiveKit's VAD (Voice Activity Detection) feature automatically detects when a user is speaking and when the AI is speaking so that the conversation flows naturally.
- **Metrics Logging:** Logs metrics to the console for monitoring and debugging.

## Setup TBD

To get this project up and running, follow these steps:

1.  **Clone the repository:**


2.  **Install dependencies:**


3.  **Environment Variables:**


## Usage TBD

To run the LiveKit agent, execute the `agent.py` script:

```bash
python agent.py
```


## Technologies Used

-   **LiveKit:** Real-time communication platform.
-   **Python:** Primary programming language.
-   **LiveKit Agents:** Framework for building AI agents on LiveKit.
-   **LiveKit Plugins:**
    -   Deepgram (Speech-to-Text)
    -   ElevenLabs (Text-to-Speech)
    -   OpenAI (Language Model)
    -   Noise Cancellation
    -   Turn Detector

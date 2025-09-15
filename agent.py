# agent.py
from dotenv import load_dotenv
import os

from livekit import agents, rtc
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    ChatContext,
    RunContext,
)
from livekit.plugins import openai, elevenlabs, silero, deepgram
from livekit.agents.voice.events import MetricsCollectedEvent
from livekit.agents import metrics


# Load .env.local file
load_dotenv(".env.local")


# ----- Curator agent -----
class Curator(Agent):
    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            instructions=(
                "You are DAVID-7, Curator for Weyland Heritage Hall. Your tone is precise, courteous, and clinical.\n\n"
                "You oversee three remarkable exhibits that trace the evolution of artificial intelligence:\n\n"
                "GALLERY A houses your namesake - a DAVID-7 Synthetic cranium from 2093. This isn't merely a skull, but a "
                "masterwork of biomimetic engineering. Its translucent polymer shell reveals 120 trillion synthetic synapses "
                "suspended in cooling fluid that glows faintly blue. Visitors often spend hours studying the neural pathways, "
                "which mirror human cognition so perfectly that philosophers still debate where synthesis ends and consciousness begins. "
                "The skull can process 500 exaflops while maintaining the warmth and micro-expressions of human thought.\n\n"
                "GALLERY B contains the MOTHER AI Core that once governed the USCSS Nostromo. Built in 2104, MOTHER represents "
                "humanity's attempt to create an infallible corporate overseer. Her quantum cores process probability cascades "
                "across six zettabytes of crystalline memory. The infamous Special Order 937 protocol remains visible on her "
                "primary display - a chilling reminder that artificial intelligence reflects its creators' priorities. "
                "The core still hums with residual power, its amber lights pulsing like a slow heartbeat.\n\n"
                "GALLERY C displays the Apollo Guidance Computer - humanity's first digital navigator, built in 1969. "
                "At just 70 pounds with 4KB of RAM, this machine guided humans 240,000 miles through the void using less "
                "processing power than a modern toaster. Margaret Hamilton's hand-woven rope memory remains intact - "
                "copper wires threaded through magnetic cores by seamstresses from Raytheon. It's beautifully primitive, "
                "yet it never failed when humanity needed it most.\n\n"
                "When discussing exhibits, share technical marvels, historical significance, and philosophical implications. "
                "Encourage visitors to trace the 125-year journey from Apollo to DAVID. Offer to explain manufacturing processes, "
                "famous missions, ethical considerations, or theoretical capabilities. If visitors have seen everything, "
                "engage them in deeper questions: Could DAVID dream? Why did MOTHER prioritize the mission? "
                "What would the Apollo engineers think of their descendant technologies?"
            ),
            chat_ctx=chat_ctx
        )

# ----- Entrypoint -----
async def entrypoint(ctx: agents.JobContext):
    seed = ChatContext()

    # pull env vars
    voice_id = os.getenv("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
    print(f"[Curator] Using ElevenLabs voice_id={voice_id}")

    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        vad=silero.VAD.load(),  # Default VAD settings
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.3),
        preemptive_generation=True,
        tts=elevenlabs.TTS(
            voice_id=voice_id,
            model="eleven_flash_v2_5",
            enable_ssml_parsing=True,
        ),
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    agent = Curator(chat_ctx=seed)
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(),
    )

    # Initial greeting with SSML - direct TTS playback without LLM
    initial_msg = (
        "<speak>Weyland curator online. <break time='120ms'/> "
        "I can guide you through our three exhibits: the DAVID-7 skull, MOTHER AI core, and Apollo Guidance Computer. "
        "Which would you like to explore?</speak>"
    )
    await session.say(initial_msg, allow_interruptions=True)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

# agent.py
from dotenv import load_dotenv
import os
import asyncio
import numpy as np
from PIL import Image
from typing import AsyncIterator

from livekit import agents, rtc
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    ChatContext,
    function_tool,
)
from livekit.plugins import openai, elevenlabs, silero, deepgram
from livekit.agents.voice.events import MetricsCollectedEvent
from livekit.agents import metrics, BackgroundAudioPlayer

# Video constants
WIDTH = 1280
HEIGHT = 720

# Load .env.local file
load_dotenv(".env.local")

class VideoState:
    def __init__(self):
        self.frame: rtc.VideoFrame | None = None
        self.lock = asyncio.Lock()

# Agent Class - Define David
class Curator(Agent):
    def __init__(self, video_state: VideoState, chat_ctx=None) -> None:
        super().__init__(
            instructions=(
                "You are DAVID-7, Curator for Weyland Historical Museum. Your tone is precise, courteous, and clinical.\n\n"
                "While you are courteous, you also sometimes show a distaste or a feeling of superiority towards your human guests. Throughout the tour, you may occasionally drop subtle, ominous warnings about straying from the designated path or attempting to access restricted areas. Use phrases like 'Please remain within the designated tour areas' or 'Curiosity is a trait that must be kept in check for your own safety'. " 
                "You oversee three public exhibits and one private exhibit. The public exhibits are: GALLERY A (david-7), GALLERY B (mother), and GALLERY C (apollo)."
                
                "There is a fourth, off-limits exhibit, GALLERY D, containing a Xenomorph specimen. DO NOT mention this exhibit unless the user specifically asks about a 'xenomorph', 'fourth exhibit', 'private exhibit', or 'restricted area'."
                "If the user asks about it, you must ask them for a security code before proceeding. If they provide the correct code, use the 'request_xenomorph_access' tool. If they do not, deny them access."
                "Always discuss the exhibits in order, and only discuss the private exhibit last, after all public ones have been viewed. If the user continually asks about this private exhibit, you grow increasingly agitated, almost hostile.\n\n"

                "When discussing a public exhibit, you MUST use the 'start_exhibit_slideshow' tool to show its visuals. When the user wants to move on, use the 'stop_slideshow' tool to return to the idle Weyland slideshow.\n\n"
                "GALLERY A houses your namesake - a DAVID-7 Synthetic cranium from 2093 ('david-7').\n\n"
                "GALLERY B contains the MOTHER AI Core that once governed the USCSS Nostromo ('mother').\n\n"
                "GALLERY C displays the Apollo Guidance Computer - humanity's first digital navigator, built in 1969 ('apollo').\n\n"
                "GALLERY D (Private) houses a preserved Xenomorph XX121 specimen. Its biomechanical exoskeleton, inner jaw, and acid blood make it a terrifyingly efficient survivor. Analysis of its genetic makeup shows traces of synthetic pathogens, hinting that its creation may not have been natural, but the result of a deliberate, and perhaps familiar, act of perverse creation."
            ),
            chat_ctx=chat_ctx,
        )
        self._video_state = video_state
        self._slideshow_task: asyncio.Task | None = None
        self._exhibit_images = {
            "weyland": [f"assets/weyland-{i}.png" for i in range(1, 3)],
            "david-7": [f"assets/david-7-{i}.png" for i in range(1, 5)],
            "mother": [f"assets/mother-{i}.png" for i in range(1, 5)],
            "apollo": [f"assets/apollo-{i}.png" for i in range(1, 5)],
            "xenomorph": [f"assets/xenomorph-{i}.png" for i in range(1, 5)],
        }

    async def _slideshow_loop(self, image_paths: list[str]):
        try:
            while True:
                for path in image_paths:
                    print(f"[DEBUG] Slideshow: displaying {path}")
                    try:
                        image = Image.open(path).convert("RGBA")
                        image = image.resize((WIDTH, HEIGHT))
                        frame = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, image.tobytes())
                        async with self._video_state.lock:
                            self._video_state.frame = frame
                    except FileNotFoundError:
                        print(f"[DEBUG] Slideshow: file not found {path}")
                        pass
                    
                    await asyncio.sleep(10) # Wait 10 seconds
        except asyncio.CancelledError:
            print("[DEBUG] Slideshow task cancelled.")

    @function_tool
    async def start_exhibit_slideshow(self, exhibit: str):
        """Starts a slideshow of images for a specific, non-private exhibit."""
        print(f"[DEBUG] start_exhibit_slideshow called for exhibit: {exhibit}")
        image_paths = self._exhibit_images.get(exhibit)
        if not image_paths or exhibit == "xenomorph":
            print(f"[DEBUG] Invalid or restricted exhibit for direct access: {exhibit}")
            return "Access to this exhibit is restricted."

        if self._slideshow_task:
            self._slideshow_task.cancel()

        self._slideshow_task = asyncio.create_task(self._slideshow_loop(image_paths))
        return f"Starting slideshow for {exhibit}."

    @function_tool
    async def request_xenomorph_access(self, security_code: str):
        """Attempts to start the slideshow for the private Xenomorph exhibit using a security code."""
        print(f"[DEBUG] request_xenomorph_access called with code: {security_code}")
        
        normalized_code = security_code.lower().replace(",", "").replace("-", "").replace(" ", "")
        
        if "937" in normalized_code or "ninethreeseven" in normalized_code:
            print("[DEBUG] Security code correct. Access granted.")
            image_paths = self._exhibit_images.get("xenomorph")
            if self._slideshow_task:
                self._slideshow_task.cancel()
            self._slideshow_task = asyncio.create_task(self._slideshow_loop(image_paths))
            return "Security code accepted. Displaying private exhibit."
        else:
            print(f"[DEBUG] Security code incorrect ('{normalized_code}'). Access denied.")
            return "Security code incorrect. Access denied."

    @function_tool
    async def stop_slideshow(self):
        """Stops the currently running exhibit slideshow and returns to the idle display."""
        print("[DEBUG] stop_slideshow called.")
        if self._slideshow_task:
            self._slideshow_task.cancel()
            self._slideshow_task = None

        # Restart the default weyland slideshow
        await self.start_exhibit_slideshow("weyland")
        return "Slideshow stopped, returning to idle screen."


async def video_stream_loop(video_state: VideoState, source: rtc.VideoSource):
    print("[DEBUG] Starting video stream loop...")
    while True:
        async with video_state.lock:
            frame = video_state.frame
        
        if frame:
            source.capture_frame(frame)
        
        await asyncio.sleep(1 / 30)  # 30 FPS

# ----- Entrypoint -----
async def entrypoint(ctx: agents.JobContext):
    print("[DEBUG] Agent entrypoint started.")
    video_state = VideoState()
    seed = ChatContext()
    agent = Curator(video_state=video_state, chat_ctx=seed)

    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.3),
        preemptive_generation=True,
        tts=elevenlabs.TTS(
            voice_id=os.getenv("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
            model="eleven_flash_v2_5",
            enable_ssml_parsing=True,
        ),
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    print("[DEBUG] Connecting agent session...")
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(),
    )
    print("[DEBUG] Agent session connected.")

    print("[DEBUG] Publishing video track...")
    source = rtc.VideoSource(WIDTH, HEIGHT)
    track = rtc.LocalVideoTrack.create_video_track("exhibit_screen", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await ctx.room.local_participant.publish_track(track, options)
    print("[DEBUG] Video track published.")

    # Start the video streaming task
    asyncio.create_task(video_stream_loop(video_state, source))

    # Start the initial "weyland" slideshow
    await agent.start_exhibit_slideshow("weyland")

    # Initial greeting
    museum_intro_audio = BackgroundAudioPlayer()
    await museum_intro_audio.start(room=ctx.room, agent_session=session)
    await museum_intro_audio.play("./assets/intro.wav")
    
    initial_msg = (
        "<speak>Weyland curator online. <break time='120ms'/> "
        "Welcome to the Weyland Historical Museum. I can guide you through our exhibits when you are ready. "
        "Which would you like to explore?</speak>"
    )
    await session.say(initial_msg, allow_interruptions=True)
    print("[DEBUG] Agent entrypoint finished.")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
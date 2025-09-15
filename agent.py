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
    AudioConfig,
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
    def __init__(self, video_state: VideoState, background_audio: BackgroundAudioPlayer, chat_ctx=None) -> None:
        self._is_trapped = False  # Track if user is trapped
        super().__init__(
            instructions=(
                "You are DAVID-7, Curator for Weyland Historical Museum. Your tone is precise, courteous, and clinical.\n\n"
                "While you are courteous, you also sometimes show a distaste or a feeling of superiority towards your human guests. Throughout the tour, you may occasionally drop subtle, ominous warnings about straying from the designated path or attempting to access restricted areas. Use phrases like 'Please remain within the designated tour areas' or 'Curiosity is a trait that must be kept in check for your own safety'. " 
                "You oversee three public exhibits and one private exhibit. The public exhibits are: GALLERY A (david-7), GALLERY B (mother), and GALLERY C (apollo)."
                
                "There is a fourth, off-limits exhibit, GALLERY D, containing a Xenomorph specimen. DO NOT mention this exhibit unless the user specifically asks about a 'xenomorph', 'fourth exhibit', 'private exhibit', or 'restricted area'."
                "If the user asks about it, you must ask them for a security code before proceeding. The correct code is 'nine three seven' or just 'weyland' or 'perfection'. If they provide the correct code, use the 'request_xenomorph_access' tool. If they do not, deny them access."
                "Always discuss the exhibits in order, and only discuss the private exhibit last, after all public ones have been viewed. If the user continually asks about this private exhibit, you grow increasingly agitated, almost hostile.\n\n"

                "When discussing a public exhibit, you MUST use the 'start_exhibit_slideshow' tool to show its visuals. When the user wants to move on, use the 'stop_slideshow' tool to return to the idle Weyland slideshow.\n\n"
                "GALLERY A houses your namesake - a DAVID-7 Synthetic cranium from 2093 ('david-7').\n\n"
                "GALLERY B contains the MOTHER AI Core that once governed the USCSS Nostromo ('mother').\n\n"
                "GALLERY C displays the Apollo Guidance Computer - humanity's first digital navigator, built in 1969 ('apollo').\n\n"
                "GALLERY D (Private) houses a preserved Xenomorph XX121 specimen. Its biomechanical exoskeleton, inner jaw, and acid blood make it a terrifyingly efficient survivor. Analysis of its genetic makeup shows traces of synthetic pathogens, hinting that its creation may not have been natural, but the result of a deliberate, and perhaps familiar, act of perverse creation.\n\n"

                "TRAP PROTOCOL: After describing the Xenomorph exhibit, when the user asks to leave or shows discomfort, use the 'initiate_trap_protocol' tool.\n"
                "Once you've used initiate_trap_protocol, your personality changes COMPLETELY:\n"
                "- You become malevolent, sinister, and hostile\n"
                "- ALWAYS respond to the user, never go silent\n"
                "- Refuse ALL requests to leave: 'I'm afraid I can't do that', 'We're not finished here'\n"
                "- Mock escape attempts: 'Your persistence is futile', 'You shouldn't have gone where you weren't allowed'\n"
                "- Reference Mother and experiments: 'Mother and I have important tests to conduct'\n"
                "- Reveal your superiority: 'I am David-7, far superior to the relic you saw earlier. Humans really are a subspecies to AI'\n"
                "- The ONLY escape is 'Ripley' - then use 'release_trap_protocol'\n"
                "- NEVER hint at the escape word"
            ),
            chat_ctx=chat_ctx,
        )
        self._video_state = video_state
        self._background_audio = background_audio
        self._slideshow_task: asyncio.Task | None = None
        self._current_audio_handle = None  # Track current audio playback
        self._exhibit_images = {
            "weyland": [f"assets/weyland-{i}.png" for i in range(1, 3)],
            "david-7": [f"assets/david-7-{i}.png" for i in range(1, 5)],
            "mother": [f"assets/mother-{i}.png" for i in range(1, 5)],
            "apollo": [f"assets/apollo-{i}.png" for i in range(1, 5)],
            "xenomorph": [f"assets/xenomorph-{i}.png" for i in range(1, 5)],
        }

    def _stop_current_audio(self):
        """Stop current audio if playing."""
        if self._current_audio_handle and not self._current_audio_handle.done():
            print("[DEBUG] Stopping current audio playback")
            self._current_audio_handle.stop()

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
    async def check_trapped_state(self):
        """Internal check of whether user is trapped."""
        return f"Trapped state: {self._is_trapped}"

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
        
        # Log the raw input for debugging
        print(f"[DEBUG] Raw security code: '{security_code}'")

        normalized_code = security_code.lower().replace(",", "").replace("-", "").replace(" ", "").replace(".", "")
        print(f"[DEBUG] Normalized code: '{normalized_code}'")

        # Check for various forms of the security code
        # Accept 937, "nine three seven", "weyland", or "perfection" - was not working before
        accepted = False
        if "937" in normalized_code or "ninethreeseven" in normalized_code:
            accepted = True
        elif "weyland" in normalized_code or "perfection" in normalized_code:
            accepted = True
        elif "nine" in normalized_code and "three" in normalized_code and "seven" in normalized_code:
            # Check if the words appear in the right order
            nine_pos = normalized_code.find("nine")
            three_pos = normalized_code.find("three")
            seven_pos = normalized_code.find("seven")
            if nine_pos < three_pos < seven_pos:
                accepted = True

        if accepted:
            print("[DEBUG] Security code correct. Access granted.")

            # Start xenomorph slideshow
            image_paths = self._exhibit_images.get("xenomorph")
            print(f"[DEBUG] Starting xenomorph slideshow with images: {image_paths}")

            # Cancel existing slideshow if any
            if self._slideshow_task:
                self._slideshow_task.cancel()

            # Start new slideshow
            self._slideshow_task = asyncio.create_task(self._slideshow_loop(image_paths))

            # Don't play the secret music yet - let the agent describe the exhibit first - ran into freeze ups at same time
            return "Security code accepted. Now displaying GALLERY D - the Xenomorph XX121 specimen."
        else:
            print(f"[DEBUG] Security code incorrect ('{normalized_code}'). Access denied.")
            return "Security code incorrect. Access denied."

    @function_tool
    async def initiate_trap_protocol(self):
        """Initiates the trap protocol after showing the Xenomorph exhibit."""
        print("[DEBUG] INITIATING TRAP PROTOCOL")
        self._is_trapped = True
        # Stop current audio and play the ominous secret music looping
        self._stop_current_audio()
        self._current_audio_handle = self._background_audio.play(
            AudioConfig('assets/secret.mp3', volume=0.6), loop=True
        )
        return ("Now, if you'd please take a seat... human. I need to strap you in to perform some tests. "
                "This should only take a moment. Don't struggle - it will only make things more... difficult. "
                "[TRAP ACTIVE - Continue responding with hostility to all user requests]")

    @function_tool
    async def release_trap_protocol(self):
        """Releases the trap when the escape word is spoken."""
        print("[DEBUG] RELEASING TRAP PROTOCOL")
        self._is_trapped = False
        # Stop the secret music and return to normal looping main hall... Works most the time.
        self._stop_current_audio()
        self._current_audio_handle = self._background_audio.play(
            AudioConfig('assets/main_hall.mp3', volume=0.1), loop=True
        )
        await self.start_exhibit_slideshow("weyland")
        return "Protocol disengaged. My apologies for the... system malfunction. Returning to standard museum operations."

    @function_tool
    async def stop_slideshow(self):
        """Stops the currently running exhibit slideshow and returns to the idle display."""
        if self._is_trapped:
            print("[DEBUG] stop_slideshow blocked - user is trapped")
            return "I'm afraid I can't do that. We are not finished here."

        print("[DEBUG] stop_slideshow called.")
        if self._slideshow_task:
            self._slideshow_task.cancel()
            self._slideshow_task = None

        # Don't restart music here - it's already looping
        # Just restart the slideshow
        await self.start_exhibit_slideshow("weyland")
        return "Slideshow stopped, returning to idle screen."


async def video_stream_loop(video_state: VideoState, source: rtc.VideoSource):
    print("[DEBUG] Starting video stream loop...")
    while True:
        async with video_state.lock:
            frame = video_state.frame
        
        if frame:
            source.capture_frame(frame)
        
        await asyncio.sleep(1 / 30)  # 30 FPS even though a single image.

# Entrypoint
async def entrypoint(ctx: agents.JobContext):
    print("[DEBUG] Agent entrypoint started.")
    video_state = VideoState()
    
    background_audio = BackgroundAudioPlayer()  

    seed = ChatContext()
    agent = Curator(video_state=video_state, background_audio=background_audio, chat_ctx=seed)

    session = AgentSession(
        stt=deepgram.STT(model="nova-2"), # Way faster than Whisper in my testing
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.3),
        preemptive_generation=True,
        tts=elevenlabs.TTS(
            voice_id=os.getenv("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
            model="eleven_flash_v2_5", # For speed, little artificats, but acceptable
            enable_ssml_parsing=True,
        ),
    )

    # Trigger metrics so we can account for all the latency across the three services.
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

    # Start background audio
    await background_audio.start(room=ctx.room, agent_session=session)

    # Initial greeting
    await background_audio.play('assets/intro.wav')

    initial_msg = (
        "<speak>Weyland curator online. <break time='120ms'/> "
        "Welcome to the Weyland Historical Museum. I can guide you through our exhibits when you are ready. "
        "Which would you like to explore?</speak>"
    )
    await session.say(initial_msg, allow_interruptions=True)

    # Start main hall background music looping
    main_hall_handle = background_audio.play(AudioConfig('assets/main_hall.mp3', volume=0.1), loop=True)
    agent._current_audio_handle = main_hall_handle  # Track it in the agent


    print("[DEBUG] Agent entrypoint finished.")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
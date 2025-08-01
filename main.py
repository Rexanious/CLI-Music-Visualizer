# ASCII Music Visualizer (Terminal)
# Requires: numpy, sounddevice, rich

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

# Settings
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024  # Number of audio frames per read
BAR_COUNT = 48      # Number of bars to display

console = Console()

# Function to get audio from microphone (can be changed to audio file with PyDub or similar)
def audio_callback(indata, frames, time, status):
    global current_audio
    if status:
        console.log(f"[bold red]Error:[/] {status}")
    current_audio = indata[:, 0]

current_audio = np.zeros(BUFFER_SIZE)

# Start audio stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE)
stream.start()

# Main visualization loop
def render_bars(audio_data):
    # Apply FFT
    fft_result = np.abs(np.fft.rfft(audio_data))
    fft_result = fft_result[:BAR_COUNT]

    # Normalize to max value (avoid divide by zero)
    max_val = np.max(fft_result)
    if max_val == 0:
        max_val = 1

    normalized = (fft_result / max_val) * 8  # Max bar height
    bars = ""

    for val in normalized:
        height = int(val)
        bars += "▇" * height + "\n"

    return Panel(bars, title="[bold cyan]ASCII Visualizer", border_style="magenta")

with Live(console=console, refresh_per_second=30):
    try:
        while True:
            audio_snapshot = np.copy(current_audio)
            panel = render_bars(audio_snapshot)
            console.print(panel)
    except KeyboardInterrupt:
        stream.stop()
        console.clear()
        console.print("[bold green]Visualizer stopped.")

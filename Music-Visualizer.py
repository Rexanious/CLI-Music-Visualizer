#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation  # THIS FIXES THE ERROR
import librosa
import os
from threading import Event


class MusicVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Visualizer")
        self.root.geometry("900x700")

        # Audio properties
        self.audio_data = None
        self.sample_rate = 44100
        self.stream = None
        self.stop_event = Event()
        self.paused = False
        self.ani = None  # Track animation object

        # UI Setup
        self.setup_ui()

    def setup_ui(self):
        # Control Frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        self.btn_open = tk.Button(
            control_frame,
            text="Open Audio File",
            command=self.open_file,
            width=15
        )
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_play = tk.Button(
            control_frame,
            text="Play",
            command=self.toggle_play,
            state=tk.DISABLED,
            width=10
        )
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(
            control_frame,
            text="Stop",
            command=self.stop_audio,
            state=tk.DISABLED,
            width=10
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Visualization Frame
        self.fig, (self.ax_wave, self.ax_fft) = plt.subplots(2, 1, figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to load audio file")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        ).pack(fill=tk.X)

    def open_file(self):
        filetypes = [
            ("Audio Files", "*.mp3 *.wav *.flac *.ogg"),
            ("All Files", "*.*")
        ]

        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            initialdir=os.path.expanduser("~/Music"),
            filetypes=filetypes
        )

        if file_path:
            try:
                self.status_var.set("Loading audio file...")
                self.root.update()

                self.audio_data, self.sample_rate = librosa.load(
                    file_path,
                    sr=None,
                    mono=True
                )

                self.file_path = file_path
                self.current_pos = 0
                self.stop_event.clear()

                self.btn_play.config(state=tk.NORMAL)
                self.status_var.set(
                    f"Loaded: {os.path.basename(file_path)} | "
                    f"Duration: {len(self.audio_data) / self.sample_rate:.2f}s"
                )

                # Initialize empty plot
                self.update_visualization()

            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                self.btn_play.config(state=tk.DISABLED)

    def toggle_play(self):
        if not self.paused:
            self.start_audio()
            self.btn_play.config(text="Pause")
        else:
            self.pause_audio()
            self.btn_play.config(text="Play")

    def start_audio(self):
        if self.audio_data is None:
            return

        self.paused = False
        self.btn_stop.config(state=tk.NORMAL)

        def audio_callback(outdata, frames, time, status):
            if self.stop_event.is_set() or self.paused:
                outdata[:] = 0
                raise sd.CallbackStop

            available = len(self.audio_data) - self.current_pos
            if available < frames:
                outdata[:available] = self.audio_data[self.current_pos:].reshape(-1, 1)
                outdata[available:] = 0
                self.current_pos += available
                raise sd.CallbackStop
            else:
                outdata[:] = self.audio_data[self.current_pos:self.current_pos + frames].reshape(-1, 1)
                self.current_pos += frames

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            finished_callback=self.audio_finished
        )

        try:
            self.stream.start()
            # THIS IS THE CRITICAL FIX - properly initialize animation
            if self.ani is not None:
                self.ani.event_source.stop()
            self.ani = FuncAnimation(
                self.fig,
                lambda i: self.update_visualization(),
                interval=50,
                cache_frame_data=False
            )
            self.canvas.draw()
        except Exception as e:
            self.status_var.set(f"Playback error: {str(e)}")

    def pause_audio(self):
        self.paused = True
        if self.stream:
            self.stream.stop()

    def stop_audio(self):
        self.stop_event.set()
        self.paused = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.ani:
            self.ani.event_source.stop()  # Properly stop animation
        self.current_pos = 0
        self.btn_play.config(text="Play")
        self.btn_stop.config(state=tk.DISABLED)
        self.update_visualization()

    def audio_finished(self):
        if not self.paused:
            self.stop_audio()

    def update_visualization(self):
        if self.audio_data is None:
            return

        chunk_size = 4096
        start_pos = max(0, self.current_pos - chunk_size)
        chunk = self.audio_data[start_pos:self.current_pos]

        if len(chunk) < 100:
            return

        self.ax_wave.clear()
        self.ax_fft.clear()

        # Waveform
        self.ax_wave.plot(chunk, color='cyan', alpha=0.8)
        self.ax_wave.set_title("Waveform")
        self.ax_wave.set_ylim(-1, 1)

        # FFT
        fft = np.abs(np.fft.rfft(chunk)[:chunk_size // 2])
        freqs = np.fft.rfftfreq(len(chunk), 1 / self.sample_rate)[:chunk_size // 2]
        self.ax_fft.plot(freqs, fft, color='magenta')
        self.ax_fft.set_title("Frequency Spectrum")
        self.ax_fft.set_xlim(0, 5000)

        # Progress info
        progress = self.current_pos / len(self.audio_data)
        self.ax_fft.text(
            0.5, -0.2,
            f"Progress: {progress:.1%} | "
            f"Status: {'Paused' if self.paused else 'Playing'}",
            transform=self.ax_fft.transAxes,
            ha='center'
        )

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicVisualizer(root)
    root.mainloop()
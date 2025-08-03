#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import librosa
import sounddevice as sd
from scipy.ndimage import gaussian_filter1d


class UltimateVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("ULTIMATE VISUALIZER")
        self.root.geometry("1200x800")
        self.root.configure(bg='#121212')

        # Audio properties
        self.audio_data = None
        self.sample_rate = 44100
        self.stream = None
        self.playing = False
        self.current_pos = 0
        self.chunk_size = 1024 * 8  # Larger chunks for better performance

        # UI colors
        self.bg_color = '#121212'
        self.fg_color = '#ffffff'
        self.accent_color = '#1f77b4'

        # Visualization settings
        self.bar_count = 50  # Default bar count
        self.smoothing = 0.7

        # Setup UI
        self.setup_ui()

        # Setup visualization
        self.fig, self.ax = plt.subplots(figsize=(12, 6), facecolor=self.bg_color)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.ax.set_facecolor(self.bg_color)

    def setup_ui(self):
        # Main control frame
        control_frame = tk.Frame(self.root, bg=self.bg_color)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Button style
        btn_style = {
            'bg': '#333333',
            'fg': self.fg_color,
            'activebackground': '#444444',
            'activeforeground': self.fg_color,
            'borderwidth': 0,
            'highlightthickness': 0,
            'padx': 15,
            'pady': 5
        }

        # File controls
        tk.Button(
            control_frame,
            text="📁 Open MP3",
            command=self.load_audio,
            **btn_style
        ).pack(side=tk.LEFT, padx=5)

        self.play_btn = tk.Button(
            control_frame,
            text="▶ Play",
            command=self.toggle_play,
            state=tk.DISABLED,
            **btn_style
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="⏹ Stop",
            command=self.stop_audio,
            **btn_style
        ).pack(side=tk.LEFT, padx=5)

        # Visualization controls
        tk.Label(
            control_frame,
            text="Bars:",
            bg=self.bg_color,
            fg=self.fg_color
        ).pack(side=tk.LEFT, padx=(20, 5))

        self.bar_slider = ttk.Scale(
            control_frame,
            from_=20,
            to=100,
            value=self.bar_count,
            command=self.update_bar_count
        )
        self.bar_slider.pack(side=tk.LEFT, padx=5)

        # Visualization type
        self.vis_type = tk.StringVar(value='wave')
        tk.Radiobutton(
            control_frame,
            text="Wave",
            variable=self.vis_type,
            value='wave',
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor=self.bg_color,
            activebackground=self.bg_color,
            command=self.update_visualization
        ).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(
            control_frame,
            text="Bars",
            variable=self.vis_type,
            value='bars',
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor=self.bg_color,
            activebackground=self.bg_color,
            command=self.update_visualization
        ).pack(side=tk.LEFT, padx=5)

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
        if file_path:
            try:
                self.audio_data, self.sample_rate = librosa.load(file_path, sr=None, mono=True)
                self.current_pos = 0
                self.play_btn.config(state=tk.NORMAL)
                self.update_visualization()
            except Exception as e:
                print(f"Error loading file: {e}")

    def toggle_play(self):
        if not self.playing:
            self.start_audio()
            self.play_btn.config(text="⏸ Pause")
        else:
            self.pause_audio()
            self.play_btn.config(text="▶ Play")

    def start_audio(self):
        self.playing = True

        def audio_callback(outdata, frames, time, status):
            if self.current_pos + frames > len(self.audio_data):
                raise sd.CallbackStop
            outdata[:, 0] = self.audio_data[self.current_pos:self.current_pos + frames]
            self.current_pos += frames

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            finished_callback=self.stop_audio
        )
        self.stream.start()

        self.ani = FuncAnimation(
            self.fig,
            self.update_visualization,
            interval=30,  # ~33fps
            cache_frame_data=False
        )
        self.canvas.draw()

    def pause_audio(self):
        self.playing = False
        if self.stream:
            self.stream.stop()
        if self.ani:
            self.ani.event_source.stop()

    def stop_audio(self):
        self.pause_audio()
        self.current_pos = 0
        self.update_visualization()
        self.play_btn.config(text="▶ Play")

    def update_bar_count(self, val):
        self.bar_count = int(float(val))
        if self.playing:
            self.update_visualization()

    def update_visualization(self, i=0):
        if self.audio_data is None:
            return

        chunk_size = 1024 * 8
        chunk = self.audio_data[self.current_pos:self.current_pos + chunk_size]

        self.ax.clear()

        if self.vis_type.get() == 'wave':
            # Waveform visualization
            self.ax.plot(chunk, color=self.accent_color, linewidth=1)
            self.ax.set_ylim(-1, 1)
        else:
            # Optimized bar visualization
            fft = np.abs(np.fft.rfft(chunk)[:chunk_size // 2])
            fft = gaussian_filter1d(fft, sigma=self.smoothing * 5)

            # Downsample to bar count
            indices = np.linspace(0, len(fft) - 1, self.bar_count, dtype=int)
            heights = fft[indices]

            # Normalize and plot
            max_height = max(1, np.max(heights))
            self.ax.bar(
                indices,
                heights / max_height,
                width=(len(fft) / self.bar_count) * 0.8,
                color=self.accent_color
            )
            self.ax.set_ylim(0, 1.1)

        # Style the plot
        self.ax.set_facecolor(self.bg_color)
        self.ax.tick_params(colors=self.fg_color)
        for spine in self.ax.spines.values():
            spine.set_color(self.fg_color)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = UltimateVisualizer(root)
    root.mainloop()
    
    
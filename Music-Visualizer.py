#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import librosa
import sounddevice as sd
from scipy.ndimage import gaussian_filter1d
import imageio
import threading
import os
import time
from queue import Queue


class Visualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("VISUALIZER")
        self.root.geometry("1200x800")
        self.root.configure(bg='#121212')

        self.audio_data = None
        self.sample_rate = 44100
        self.stream = None
        self.playing = False
        self.current_pos = 0
        self.chunk_size = 1024 * 15

        self.recording = False
        self.frames = []
        self.export_queue = Queue()
        self.export_thread = None
        self.export_progress = 0
        self.export_total_frames = 0

        self.bar_count = 50
        self.smoothing = 0.8
        self.vis_type = "bars"
        self.color_scheme = "magenta"

        self.setup_ui()

        self.fig, self.ax = plt.subplots(figsize=(12, 6), facecolor='#121212')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_facecolor('#121212')

    def setup_ui(self):
        control_frame = tk.Frame(self.root, bg='#121212')
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Control buttons
        btn_style = {
            'bg': '#333333',
            'fg': 'white',
            'activebackground': '#444444',
            'borderwidth': 0,
            'highlightthickness': 0,
            'padx': 15,
            'pady': 5
        }

        tk.Button(
            control_frame,
            text="Open MP3",
            command=self.load_audio,
            **btn_style
        ).pack(side=tk.LEFT, padx=5)

        self.play_btn = tk.Button(
            control_frame,
            text="Play",
            command=self.toggle_play,
            state=tk.DISABLED,
            **btn_style
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="Stop",
            command=self.stop_audio,
            **btn_style
        ).pack(side=tk.LEFT, padx=5)

        self.export_btn = tk.Button(
            control_frame,
            text="Export Video",
            command=self.toggle_recording,
            state=tk.DISABLED,
            **btn_style
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)

        self.progress_label = tk.Label(
            control_frame,
            text="",
            bg='#121212',
            fg='white'
        )
        self.progress_label.pack(side=tk.LEFT, padx=10)

        tk.Label(
            control_frame,
            text="Bars:",
            bg='#121212',
            fg='white'
        ).pack(side=tk.LEFT, padx=(20, 5))

        self.bar_slider = ttk.Scale(
            control_frame,
            from_=20,
            to=100,
            value=self.bar_count,
            command=self.update_bar_count
        )
        self.bar_slider.pack(side=tk.LEFT, padx=5)

        self.vis_type_var = tk.StringVar(value="bars")
        tk.Radiobutton(
            control_frame,
            text="Wave",
            variable=self.vis_type_var,
            value="wave",
            bg='#121212',
            fg='white',
            selectcolor='#121212',
            command=self.change_vis_type
        ).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(
            control_frame,
            text="Bars",
            variable=self.vis_type_var,
            value="bars",
            bg='#121212',
            fg='white',
            selectcolor='#121212',
            command=self.change_vis_type
        ).pack(side=tk.LEFT, padx=5)

        self.color_var = tk.StringVar(value="magenta")
        color_options = ["magenta", "cyan", "rainbow", "green", "blue"]
        tk.OptionMenu(
            control_frame,
            self.color_var,
            *color_options,
            command=self.change_color
        ).pack(side=tk.LEFT, padx=5)

    def change_color(self, *args):
        self.color_scheme = self.color_var.get()
        self.update_visualization()

    def change_vis_type(self):
        self.vis_type = self.vis_type_var.get()
        self.update_visualization()

    def update_bar_count(self, val):
        self.bar_count = int(float(val))
        self.update_visualization()

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
        if file_path:
            try:
                self.audio_data, self.sample_rate = librosa.load(file_path, sr=None, mono=True)
                self.current_pos = 0
                self.play_btn.config(state=tk.NORMAL)
                self.export_btn.config(state=tk.NORMAL)
                self.update_visualization()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio file:\n{e}")

    def toggle_play(self):
        if not self.playing:
            self.start_audio()
            self.play_btn.config(text="Pause")
        else:
            self.pause_audio()
            self.play_btn.config(text="Play")

    def start_audio(self):
        self.playing = True

        def audio_callback(outdata, frames, time, status):
            if self.current_pos + frames > len(self.audio_data):
                outdata[:] = 0
                raise sd.CallbackStop
            outdata[:, 0] = self.audio_data[self.current_pos:self.current_pos + frames]
            self.current_pos += frames

        try:
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
                cache_frame_data=False,
                blit=False
            )
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Audio Error", f"Failed to start audio stream:\n{e}")
            self.playing = False
            self.play_btn.config(text="Play")

    def pause_audio(self):
        self.playing = False
        if self.stream:
            self.stream.stop()
        if hasattr(self, 'ani') and self.ani:
            self.ani.event_source.stop()

    def stop_audio(self):
        self.pause_audio()
        self.current_pos = 0
        self.update_visualization()
        self.play_btn.config(text="Play")

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.export_progress = 0
        self.play_btn.config(state=tk.DISABLED)
        self.export_btn.config(text="Stop Recording")
        self.progress_label.config(text="Recording...")

    def stop_recording(self):
        self.recording = False
        self.export_total_frames = len(self.frames)
        self.export_thread = threading.Thread(target=self.export_video, daemon=True)
        self.export_thread.start()
        self.update_progress()

    def update_progress(self):
        if self.export_thread and self.export_thread.is_alive():
            progress = (self.export_progress / self.export_total_frames) * 100
            self.progress_label.config(text=f"Exporting: {progress:.1f}%")
            self.root.after(100, self.update_progress)
        else:
            self.progress_label.config(text="")
            self.play_btn.config(state=tk.NORMAL)
            self.export_btn.config(text="Export Video")

    def export_video(self):
        if not self.frames:
            messagebox.showwarning("Export", "No frames recorded to export!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )

        if not file_path:
            return

        try:
            with imageio.get_writer(file_path, fps=30) as writer:
                for i, frame in enumerate(self.frames):
                    writer.append_data(frame)
                    self.export_progress = i + 1
            messagebox.showinfo("Export Complete", f"Successfully exported:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export video:\n{e}")
        finally:
            self.frames = []  # Clear frames to save memory

    def get_color(self, index, total):
        if self.color_scheme == "magenta":
            return 'magenta'
        elif self.color_scheme == "cyan":
            return 'cyan'
        elif self.color_scheme == "green":
            return '#00ff00'
        elif self.color_scheme == "blue":
            return '#0000ff'
        elif self.color_scheme == "rainbow":
            hue = index / total
            return plt.cm.hsv(hue)

    def update_visualization(self, i=0):
        if self.audio_data is None:
            return

        if self.current_pos >= len(self.audio_data):
            self.stop_audio()
            return

        chunk = self.audio_data[self.current_pos:self.current_pos + self.chunk_size]
        if len(chunk) == 0:
            return

        self.ax.clear()

        if self.vis_type == "wave":
            color = self.get_color(0, 1)
            self.ax.plot(chunk, color=color)
            self.ax.set_ylim(-1, 1)
        else:

            fft = np.abs(np.fft.rfft(chunk)[:self.chunk_size // 2])
            fft = gaussian_filter1d(fft, sigma=self.smoothing * 5)


            indices = np.linspace(0, len(fft) - 1, self.bar_count, dtype=int)
            heights = fft[indices]
            max_height = max(1, np.max(heights))

            colors = [self.get_color(i, self.bar_count) for i in range(self.bar_count)]
            self.ax.bar(
                indices,
                heights / max_height,
                width=(len(fft) / self.bar_count) * 0.8,
                color=colors
            )
            self.ax.set_ylim(0, 1.1)


        self.ax.set_facecolor('#121212')
        self.ax.set_xticks([])
        self.ax.set_yticks([])


        if self.recording:
            self.fig.canvas.draw()

            img = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            img = img[:, :, :3]
            self.frames.append(img)

        return self.ax.artists if self.vis_type == "bars" else []


if __name__ == "__main__":
    root = tk.Tk()
    app = Visualizer(root)
    root.mainloop()
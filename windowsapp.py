import tkinter as tk
import tensorflow as tf
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import librosa
import librosa.display
import pyaudio
import wave
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import *
from matplotlib.figure import Figure
from live_audiocnn_graph import *
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

root = tk.Tk()
root.wm_title("Audio CNN")
fig = Figure(figsize=(4, 4), dpi=140)
ax = fig.add_subplot(1,1,1)

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.get_tk_widget().grid(row=0, column=0)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=0)
plt.style.use('seaborn-pastel')
w = tk.Scale(root, from_=1, to=10, orient = tk.HORIZONTAL, label = "Time")
w.pack()
v = tk.StringVar()
c = tk.Label(root, textvariable = v).pack()
RECORD_SECONDS = 1
def Start():
    global RECORD_SECONDS
    RECORD_SECONDS = w.get()
    v.set("Running")
    anim.event_source.start()


def End():
    v.set("Stopped")
    anim.event_source.stop()



def itergraph(i):

    audio = pyaudio.PyAudio()
    audio.get_default_input_device_info()
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    #print("recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    #print("finished recording")
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    classpreds = print_prediction('cache_audio.wav')
    ax.clear()
    ax.bar(classpreds[0], classpreds[1], align='center')


a = tk.Button(root, text ="StartAnimation", command = Start)
a.pack()
b = tk.Button(root, text ="StopAnimation", command = End)
b.pack()
anim = FuncAnimation(fig, itergraph,
                               interval=20)


anim.event_source.stop()

root.mainloop()

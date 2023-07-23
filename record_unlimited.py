import sounddevice as sd
import numpy as np
import queue
import tempfile
import os
import ffmpeg
import soundfile as sf
import threading

# Define the silence duration in seconds
silence_duration = 2  # 2 seconds

# Define the sample rate
samplerate = 48000

# Global variables for thresholds and recording status
silence_threshold = 1
speak_threshold = 4
recording_stopped = threading.Event()
is_recording_started = threading.Event()

def calibrate():
    global silence_threshold, speak_threshold

    print("Please be silent. We are setting the silence threshold...")
    input("Press Enter when ready...")
    with sd.InputStream(channels=1, samplerate=samplerate) as stream:
        silence_threshold = np.mean(np.array([np.linalg.norm(stream.read(1000)[0])*10 for _ in range(10)]))
    print(f"Silence threshold set to: {silence_threshold}")

    print("Please speak. We are setting the speaking threshold...")
    input("Press Enter when ready...")
    with sd.InputStream(channels=1, samplerate=samplerate) as stream:
        speak_threshold = np.mean(np.array([np.linalg.norm(stream.read(1000)[0])*10 for _ in range(10)]))
    print(f"Speaking threshold set to: {speak_threshold}")

def callback(indata, frames, time, status):
    global silence_time, q, silence_threshold, speak_threshold, recording_stopped, is_recording_started
    volume_norm = np.linalg.norm(indata)*10
    if volume_norm > speak_threshold:
        q.put(indata.copy())
        silence_time = 0
        is_recording_started.set()
    elif silence_time > silence_duration and is_recording_started.is_set():
        recording_stopped.set()
    elif is_recording_started.is_set():
        silence_time += frames / samplerate

def record_unlimited() -> np.ndarray:
    global silence_time, q, silence_threshold, speak_threshold, recording_stopped, is_recording_started
    silence_time = 0
    q = queue.Queue()
    recording_stopped.clear()
    is_recording_started.clear()

    filename = tempfile.mktemp(prefix="delme_rec_unlimited_", suffix=".wav", dir="")
    try:
        with sf.SoundFile(filename, mode='x', samplerate=samplerate, channels=1) as file:
            with sd.InputStream(callback=callback, channels=1, samplerate=samplerate):
                print("Start talking...")
                while not recording_stopped.is_set():
                    try:
                        file.write(q.get(timeout=silence_duration))
                    except queue.Empty:
                        if is_recording_started.is_set():
                            print('Recording finished: ' + repr(filename))
                            break
    except Exception as e:
        print(f'Error occurred: {str(e)}')

    y, _ = ffmpeg.input(os.path.abspath(filename), threads=0).output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000).run(capture_stdout=True, capture_stderr=True)

    os.remove(filename)
    return np.frombuffer(y, np.int16).flatten().astype(np.float32) / 32768.0

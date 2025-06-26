import time, psutil
from stt import Model
import wave
import numpy as np

start_time = time.time()
model = Model("models/coqui/model.tflite")
wf = wave.open("samples/harvard.wav", "rb")

frames = wf.readframes(wf.getnframes())
audio = np.frombuffer(frames, np.int16)
text = model.stt(audio)

end_time = time.time()
print("Text:", text)
print("Time:", end_time - start_time, "seconds")
print("Memory:", psutil.Process().memory_info().rss / 1024 ** 2, "MB")


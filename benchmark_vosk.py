import wave, time, psutil
from vosk import Model, KaldiRecognizer
import json

start_time = time.time()
model = Model("models/vosk/vosk-model-small-en-us-0.15")
wf = wave.open("samples/harvard.wav", "rb")
rec = KaldiRecognizer(model, wf.getframerate())

text = ""
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        text += json.loads(rec.Result())["text"] + " "
text += json.loads(rec.FinalResult())["text"]

end_time = time.time()
print("Text:", text)
print("Time:", end_time - start_time, "seconds")
print("Memory:", psutil.Process().memory_info().rss / 1024 ** 2, "MB")


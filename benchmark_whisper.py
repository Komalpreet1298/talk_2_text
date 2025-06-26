import time, psutil, whisper

start_time = time.time()
model = whisper.load_model("small")
result = model.transcribe("samples/harvard.wav", task="translate")
end_time = time.time()

print("Text:", result["text"])
print("Time:", end_time - start_time, "seconds")
print("Memory:", psutil.Process().memory_info().rss / 1024 ** 2, "MB")


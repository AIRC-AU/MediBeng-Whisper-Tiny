import requests

with open("D:\\pycharmcode\\jiqifanyi end\\MediBeng-Whisper-Tiny-main\\tests\\data\\Male-Bengali-English-1959.wav", "rb") as f:
    files = {
        "file": ("Female-Bengali-English-2072.wav", f, "audio/wav")
    }
    response = requests.post("http://127.0.0.1:8000/transcribe", files=files)

print(response.json())

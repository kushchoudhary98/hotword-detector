import pyaudio

CHUNK=16000
FORMAT=pyaudio.paFloat32
CHANNELS=1
RATE=16000

import time

p=pyaudio.PyAudio()

stream=p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
    )

frames=[]
print("starting")
time.sleep(2)
print("started")

for i in range(0,1):
    data=stream.read(CHUNK)
    frames.append(data)
print("ended")
#print(frames)
#print(len(frames[0]))
import wave
wf=wave.open("training_/jarvis/1.wav","wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(len(frames[0]))
maxi=max(x for x in frames[0])
print(frames[0][0])
print("maxi : "+str(maxi))
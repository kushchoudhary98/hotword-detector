from tensorflow.keras.models import load_model

model=load_model("jarvis.h5")

import pyaudio

CHUNK=1024
FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=16000

p=pyaudio.PyAudio()
stream=p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
    )


import numpy as np

import time

while True:
    frames=[]
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("Speak...")
    for i in range(0,int(RATE/CHUNK*1.3)):
        data=stream.read(CHUNK)
        data=[int(x) for x in data]
        frames.extend(data)
    print("Analysing...")
    frame=np.array(frames)
    print(frame.shape)
    print(frame[:100])
    frame=frame/255
    frame=frame.reshape(1,40960)
    p=model.predict(frame)
    print(p)
    if p[0][1]>0.5:
        print("jarvis")
    else:
        print("no")
    
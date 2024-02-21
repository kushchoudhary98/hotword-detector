'''
import librosa

dirr="training_/jarvis/5.wav"
audio,sr=librosa.load(dirr,sr=16000)
mfcc=librosa.feature.mfcc(y=audio,sr=16000)

import matplotlib.pyplot as plt
plt.plot(mfcc)
#librosa.display.specshow(mfcc, x_axis='time')
plt.show()

print(audio.shape)
print(audio)
print(sr)


'''
'''
import pyaudio

CHUNK=8000
FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=16000



import wave
import numpy as np
def save_audio():
    p=pyaudio.PyAudio()
    stream=p.open(
    format=FORMAT,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
    )
    frames=[]
    print("start")
    for _ in range(1):
        data=stream.read(CHUNK)
        #data=[int(x) for x in data]
        frames.extend(data)
    frames=np.array(frames)
    print(frames)
    print(frames.shape)
    wf = wave.open("training_/jarvis/5.wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == "__main__":
    save_audio()
'''
from tensorflow.keras.models import load_model

model=load_model("marvin.h5")

from playsound import playsound

#import t2
import numpy as np
#import speech_recognition
import librosa

playsound("ding.wav")

def predict(audio_data):
    audio=audio_data.reshape(16000)
    filtered=librosa.feature.mfcc(y=audio,sr=16000)
    data=[]
    mfccScaled=[]
    for i in filtered:
        mfcc=[float(x) for x in i]
        mfccScaled.append(mfcc)
    data.append(mfccScaled)
    #data=np.expand_dims(data,axis=0)
    data=np.expand_dims(data,axis=3)
    p=model.predict(data)
    #print(p[0])
    if p[0][0]>0.9:
        print("yes, sir")
        playsound("dong.wav")

def st():
    import t2
    while True:
        audio=t2.audio_save()
        predict(audio)
    
if __name__ == "__main__":
    import threading
    t1=threading.Thread(target=st)
    t2=threading.Thread(target=st)
    import time
    t1.start()
    time.sleep(0.5)
    t2.start()

'''
import pyaudio
import time
import wave

p=pyaudio.PyAudio()

CHUNK=2048
FRAME_RATE=16000
FORMAT=pyaudio.paInt32
CHANNEL=1

stream=p.open(
    rate=FRAME_RATE,
    channels=CHANNEL,
    format=FORMAT,
    input=True
    )
wf=wave.open("training/positive-identification/1.wav","rb")
audio=wf.readframes(55555555)
audio=[float(x) for x in audio]
print(audio[:100])
print(len(audio))
import numpy as np
import librosa
audio=np.array(audio,dtype=np.float)
audio=librosa.feature.mfcc(audio,sr=16000)
print(audio.shape)
import matplotlib.pyplot as plt
plt.plot(audio)
plt.show()
'''
'''
import numpy as np
import librosa
audio,sr=librosa.load("ttt.wav",sr=16000)

print(audio[:100])
print(len(audio))
audio=np.array(audio,dtype=np.float)
audio=librosa.feature.mfcc(audio,sr=16000)
import matplotlib.pyplot as plt
plt.plot(audio)
plt.show()
'''
'''
import tensorflow as tf
import time
import wave


class myModel(tf.keras.models.Model):
    def __init__(self):
        super(myModel,self).__init__()
        self.conv1=tf.keras.layers.Conv2D(64,(3,3),activation="relu")
        self.conv2=tf.keras.layers.Conv2D(32,(3,3),activation="relu")
        self.pool1=tf.keras.layers.MaxPool2D()
        self.pool2=tf.keras.layers.MaxPool2D()
        self.gru=tf.keras.layers.GRU(20,dropout=0.3)
        self.dense1=tf.keras.layers.Dense(10,activation="relu")
        self.out=tf.keras.layers.Dense(1,activation="sigmoid")
    def call(self,x):
        x=self.gru(x)
        x=self.dense1(x)
        x=self.out(x)
        return x
        
model=tf.keras.models.load_model("saved_model/new_marvin")

wf=wave.open("training/positive-identification/1.wav","rb")
audio=wf.readframes(55555555)
audio=[float(x) for x in audio]
print(audio[:100])
print(len(audio))
import numpy as np
import librosa
audio=np.array(audio,dtype=np.float)
audio=librosa.feature.mfcc(audio,sr=16000)
print(audio.shape)

#audio=np.expand_dims(audio,axis=0)
audio=np.expand_dims(audio,axis=0)

out=model(audio)
print(out.shape)
print(out)
'''
'''
import sounddevice as sd
import soundfile as sf

samplerate = 16000  
duration = 1 # seconds
filepath="training/negative-identification/"
for i in range(200):
    filename=filepath+str(i)+".wav"
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
        channels=1, blocking=True)
    sd.wait()
    sf.write(filename, mydata, samplerate)
import requests

url = "https://google-search3.p.rapidapi.com/api/v1/search/q%253Delon%252Bmusk%2526num%253D100"

headers = {
    'x-rapidapi-host': "google-search3.p.rapidapi.com",
    'x-rapidapi-key': "8c62825cadmsh8529bac14a8551dp192c32jsndf033469d419"
    }

response = requests.request("GET", url, headers=headers)

print(response.json)
'''
import tensorflow as tf
import numpy as np

layer=tf.keras.layers.Dense(256)
x=np.zeros((1,64,2048))
output=layer(x)
print(output.shape)
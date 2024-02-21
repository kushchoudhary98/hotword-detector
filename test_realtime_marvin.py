from tensorflow.keras.models import load_model

model=load_model("marvin.h5")

from playsound import playsound

import t2
import numpy as np
#import speech_recognition
import librosa
#dirr="training_/jarvis/yes.wav"
#audio1,sr=librosa.load(dirr,sr=16000)
playsound("ding.wav")
while True:
    #print("Speak...")
    audio=t2.audio_save()
    #dirr="training_/jarvis/yes.wav"
    #audio,sr=librosa.load(dirr,sr=16000)
    audio=audio.reshape(16000)
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
    print(p[0])
    if p[0][0]>0.96:
        print("yes, sir")
        playsound("dong.wav")
    
import librosa
import os
import numpy as np

TEST_DIR_1="testing/negative/"

s_r=16000

jarvis=os.listdir(TEST_DIR_1)

test_data=[]
'''
for file_name in jarvis:
    f=TEST_DIR_1+file_name
    audio,sample_rate=librosa.load(f,sr=s_r)
    audio=np.array(audio)
    filtered=librosa.feature.mfcc(y=audio,sr=s_r)
    mfccScaled=[]
    for i in filtered:
        mfcc=[float(x) for x in i]
        mfccScaled.append(mfcc)
    test_data.append(mfccScaled)
'''
audio,sample_rate=librosa.load("training_/jarvis/yes.wav",sr=s_r)
audio=np.array(audio)
filtered=librosa.feature.mfcc(y=audio,sr=s_r)
mfccScaled=[]
for i in filtered:
    mfcc=[float(x) for x in i]
    mfccScaled.append(mfcc)
test_data.append(mfccScaled)

test_data=np.expand_dims(test_data,axis=3)
print(test_data.shape)

from tensorflow.keras.models import load_model

model=load_model("marvin.h5")

p=model.predict(test_data)

print(p)
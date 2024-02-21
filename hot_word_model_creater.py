import librosa
import os
import numpy as np

TRAIN_DIR_1="training/positive-identification/"
TRAIN_DIR_2="training/negative-identification/"

s_r=16000

jarvis=os.listdir(TRAIN_DIR_1)
other=os.listdir(TRAIN_DIR_2)


train_data=[]
train_label=[]

for file_name in jarvis:
    f=TRAIN_DIR_1+file_name
    audio,sample_rate=librosa.load(f,sr=s_r)
    audio=np.array(audio)
    filtered=librosa.feature.mfcc(y=audio,sr=s_r)
    mfccScaled=[]
    for i in filtered:
        mfcc=[float(x) for x in i]
        mfccScaled.append(mfcc)
    train_data.append(mfccScaled)
    train_label.append(1)

for file_name in other:
    f=TRAIN_DIR_2+file_name
    audio,sample_rate=librosa.load(f,sr=s_r)
    filtered=librosa.feature.mfcc(y=audio,sr=s_r)
    mfccScaled=[]
    for i in filtered:
        mfcc=[float(x) for x in i]
        mfccScaled.append(mfcc)
    train_data.append(mfccScaled)
    train_label.append(0)

train_data=np.array(train_data)
train_label=np.array(train_label)

print(train_data[0].shape)
print(train_data[0][0][:100])

train_data=np.expand_dims(train_data,axis=-1)
print(train_data.shape)
import tensorflow as tf
#train_data=tf.convert_to_tensor(train_data)

#train_data=np.array(train_data)


print("creating model")
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),input_shape=(20,32,1)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPool2D((3,3)),
    tf.keras.layers.Conv2D(64,(1,1)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32,activation="relu"),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.summary()

import sys

print("Want to continue?")
n=input("")
if n=="no":
    sys.exit(0)

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
print("starting training")
try:
    model.fit(train_data,train_label,epochs=100)
except KeyboardInterrupt:
    model.save("marvin.h5")
model.save("marvin.h5")

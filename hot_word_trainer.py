import librosa
import os
import numpy as np
import tensorflow as tf

TRAIN_DIR_1="training/positive-identification/"
TRAIN_DIR_2="training/negative-identification/"

s_r=16000

hot_word=os.listdir(TRAIN_DIR_1)
other=os.listdir(TRAIN_DIR_2)


train_data=[]
train_label=[]
import wave

for file_name in hot_word:
    f=TRAIN_DIR_1+file_name
    wf=wave.open(f,"rb")
    audio=wf.readframes(55555555)
    audio=[float(x) for x in audio]
    audio=np.array(audio,dtype=np.float)
    mfcc=librosa.feature.mfcc(audio,sr=16000)
    train_data.append(mfcc)
    train_label.append(1)

for file_name in other:
    f=TRAIN_DIR_2+file_name
    wf=wave.open(f,"rb")
    audio=wf.readframes(55555555)
    audio=[float(x) for x in audio]
    audio=np.array(audio,dtype=np.float)
    mfcc=librosa.feature.mfcc(audio,sr=16000)
    train_data.append(mfcc)
    train_label.append(0)

#train_data=np.expand_dims(train_data,axis=-1)
train_data=np.expand_dims(train_data,axis=1)
train_label=np.array(train_label)
#dataset=tf.data.Dataset.from_tensor_slices((train_data,train_label)).shuffle(10)

print("creating model")
class myModel(tf.keras.models.Model):
    def __init__(self):
        super(myModel,self).__init__()
        self.gru=tf.keras.layers.GRU(20,dropout=0.3)
        self.out=tf.keras.layers.Dense(1,activation="sigmoid")
    def call(self,x):
        #x=self.conv1(x)
        #x=self.pool1(x)
        #x=self.conv2(x)
        #x=self.pool2(x)
        #x=tf.reshape(x,(x.shape[0],-1,x.shape[-1]))
        x=self.gru(x)
        x=self.out(x)
        return x
        
model=myModel()

#model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
#loss_object=tf.keras.losses.binary_crossentropy(from_logits=True)
optimizer_object=tf.keras.optimizers.RMSprop()
t_loss=tf.keras.metrics.Mean(name="training_loss")
accuracy=tf.keras.metrics.BinaryAccuracy(name="accuracy")
#model.summary()

import sys

print("Want to continue?")
n=input("")
if n=="no":
    sys.exit(0)


print("starting training")
EPOCH=20

def train_step(data,label):
    with tf.GradientTape() as tape:
        prediction=model(data,training=True)
        loss=tf.keras.losses.binary_crossentropy(label,prediction)

        gradients=tape.gradient(loss,model.trainable_variables)
        optimizer_object.apply_gradients(zip(gradients,model.trainable_variables))

        t_loss(loss)
        accuracy(label,prediction)


from tqdm import tqdm

try:
    for epoch in range(EPOCH):
        print("Epoch "+str(epoch))

        t_loss.reset_states()
        accuracy.reset_states()

        length=len(train_data)
        for l in tqdm(range(length)):
            train_step(train_data[l],train_label[l])
    
        print("loss "+str(t_loss.result()))
        print("accuracy "+str(accuracy.result()*100)+"%")
    
except KeyboardInterrupt:
    model.save("saved_model/new_marvin")

        

model.save("saved_model/new_marvin")


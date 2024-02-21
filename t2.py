import sounddevice as sd
import soundfile as sf

def audio_save():
    samplerate = 16000  
    duration = 1 # seconds
    #filename = 'training_/jarvis/yes.wav'
    #print("start")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
        channels=1, blocking=True)
    #print("end")
    return mydata
    #sd.wait()
    #sf.write(filename, mydata, samplerate)

def test():
    import t
    while True:
        samplerate = 16000  
        duration = 1 # seconds
        #filename = 'training_/jarvis/yes.wav'
        #print("start")
        mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
            channels=1, blocking=True)
        t.predict(mydata)


if __name__ == "__main__":
    audio_save()
    '''
    import threading
    t1=threading.Thread(target=test)
    t2=threading.Thread(target=test)
    import time
    t1.start()
    time.sleep(0.5)
    t2.start()
    '''
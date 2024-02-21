from gtts import gTTS

text="Todays Headlines are. Tested Positive For Coronavirus, Hospitalised, Tweets Amit Shah. Amitabh Bachchan tests negative for Covid-19, discharged from hospital, confirms Abhishek Bachchan. Union home minister Amit Shah tests positive for COVID-19, admitted to hospital"
audio=gTTS(text)
audio.save("hi.mp3")
import playsound
playsound.playsound("hi.mp3")

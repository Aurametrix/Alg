import speech_recognition as sr
import pywapi
from gtts import gTTS

while True:
    a = sr.Recognizer()
    with sr.Microphone() as source:
        print("say something")
        audio=a.listen(source)

    try:
        print("You said: " + a.recognize_google(audio)) # change it when you finish the program.
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        continue
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    w_inputs=["get me the weather","what is the weather","tell me the weather","tell me the weather please"]
    w1_inputs=["how is the weather tommorow?"]
    h_inputs=["hello friend","hey zepo","What's up zepo?","hello"]
    s_inputs=["shutdown","stop","thank you stop","please stop"]
    if a.recognize_google(audio) in w_inputs:
        weather = pywapi.get_weather_from_weather_com("DAXX0009")
        print("Weather.com says: It is " + weather['current_conditions']['text'].lower() + " and " +
              weather['current_conditions']['temperature'] + "°C ")


    elif a.recognize_google(audio) in h_inputs:
     print("Hello friend ! ")
    elif a.recognize_google(audio)in s_inputs:
        print("Okay sir,as you wish.")
        break

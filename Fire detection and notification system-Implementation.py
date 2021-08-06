#Import the necessary packages
import os
import cv2
import emoji
import smtplib
import requests
from playsound import playsound
import numpy as np
import pywhatkit as wp
import multiprocessing
from datetime import datetime
from keras.models import load_model
import geocoder
from geopy.geocoders import Nominatim

# Setting the environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# function to make prediction
def make_prediction(image, model, class_dictionary):
    img = image / 255.
    
    # convert to 4D tensor
    image = np.expand_dims(img, axis=0)

    # train
    class_predicted = model.predict(image)
    inID = np.argmax(class_predicted[0])
    label = class_dictionary[inID]
    return label

# function for loading the weights
weights_path = 'C:/Users/hephzibah/PROJECTS/Miniproject/Fire Recognition/fire'
def keras_model(weights_path):
    model = load_model(weights_path)
    return model

# initializing the classes
class_dictionary = {}
class_dictionary[0] = 'fire'
class_dictionary[1] = 'not a fire'
model = keras_model(weights_path)

# initialising necessary data structure and flags
queue = [0,0,0]
fire_reported = 0
alarm_status = False
email_status = False

# Getting the location of the fire accident
geolocator = Nominatim(user_agent="geoapiExercises")
res = requests.get('https://ipinfo.io/')
data = res.json()
loc = data['loc'].split(',')
lat = loc[0]
long = loc[1]
location = geolocator.reverse(lat+","+long)
  
# Address stored in message string
message = "Fire Accident occured at ABC Company.\nLocation:\nAddress:"+str(location)+"\nLatitude:"+lat+"\nLongitude:"+long

# function to send notification through email and whatsapp 
def notify():
    to_email = "receiver's_mail_id"
    to_email = to_email.lower()

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login('your_mail_id','your_mail_password')
        server.sendmail('your_mail_id', to_email,message)
        print("Email sent to " + to_email)
        server.close()
        wp.sendwhatmsg("+919766712345",emoji.emojize(":loudspeaker::loudspeaker: :loudspeaker:\n\nFire :fire: occured at ABC company.\nLocation:")+str(location)+"\ \nLATITUDE : " + lat + "\tLONGITUDE : " + long + emoji.emojize("\n\n:police_car_light::police_car_light::police_car_light:"),datetime.now().hour,datetime.now().minute+1,4)    
        print("Whatsapp message sent!!")
    except Exception as e:
        print(e)

# Starting the video capture
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (200, 200))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Mask making
    l_m = np.array([0, 120, 200])
    u_m = np.array([50, 250, 250])
    mask = cv2.inRange(hsv, l_m, u_m)

    # Image morphology operation
    kernel1 = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    img = frame.copy()
    ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        l = cv2.arcLength(cnt, True)
        if l > 51:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # CNN data input
            img_test = frame[y:y + h, x:x + w]
            img_test = cv2.resize(img_test, (224, 224))
            label = make_prediction(img_test, model, class_dictionary)
            
            if label == 'fire':
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, "Fire", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                area = (x+w) * (y+h)
                if len(queue) == 3:
                    queue.pop(0)
                queue.append(area)
                if queue[0] < queue[1] < queue[2] and 0 not in queue:
                    fire_reported = 1
                    if fire_reported == 1:    
                        if alarm_status == False and email_status == False:
                            multiprocessing.Process(target=playsound, args=("Fire Alarm.mp3",)).start()  # Ringing the fire alarm 
                            alarm_status = True
                            multiprocessing.Process(target=notify()).start() # Starting the notification function
                            email_status = True     
                        
            else:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Not a Fire", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cv2.destroyAllWindows()
cap.release()






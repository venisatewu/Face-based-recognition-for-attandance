from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
import xlwt
import xlwrite
import time
import sys

app= Flask(__name__)


work_dir = "D:/Venisa/Summer/ComVis/Project/Face_recog_attendance";

#Take all facial samples on dataset directory, returning 2 arrays
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(work_dir+'/trainer/trainer.yml')


faceCascade = cv2.CascadeClassifier("{0}/haarcascade_frontalface_default.xml".format(work_dir))
font = cv2.FONT_HERSHEY_SIMPLEX

# Set Current Working Directory
camera = cv2.VideoCapture(0)

#iniciate id counter
id = 0

filename = 'filename'
dict = {
        'item' : 1
        }

def gen_frames():  
    while True:
        success, frame = camera.read()  #read the camera frame
        frame = cv2.flip(frame, 1) # Video rotation
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if not success:
            break
        else:
            
            faces = faceCascade.detectMultiScale(gray,1.1,7)
            
            #Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                
                if (confidence < 100):
                    if (id==1):
                        id = "Tewu, Venisa Octriane"
                        if ((str(id)) not in dict):
                            filename = xlwrite.output('attendance', 'class1', 1, id, 'yes');
                            dict[str(id)] = str(id);
                            
                    elif (id==2):
                        id = "Rampengan, Elshadai"
                        if ((str(id)) not in dict):
                            filename = xlwrite.output('attendance', 'class1', 2, id, 'yes');
                            dict[str(id)] = str(id);
                            
                    elif (id==3):
                        id = "Tumewu, Mawar"
                        if ((str(id)) not in dict):
                            filename = xlwrite.output('attendance', 'class1', 3, id, 'yes');
                            dict[str(id)] = str(id);
                    
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                    
                cv2.putText(
                        frame, 
                        str(id), 
                        (x+5,y-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                        )
                cv2.putText(
                        frame, 
                        str(confidence), 
                        (x+5,y+h-5), 
                        font, 
                        1, 
                        (255,255,0), 
                        1
                        )  
             
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)





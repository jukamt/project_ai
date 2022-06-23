from fileinput import filename
from time import sleep
from tensorflow.keras.models import load_model
import numpy as np
import os
from sklearn.svm import LinearSVC
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
import numpy as np

import cv2
from tkinter import *
from tkinter.filedialog import askopenfile, askopenfilename

from PIL import ImageTk,Image

emotion_model=load_model("model/emotion_model.h5")
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)



width, height = 720, 480
cap.set(3, width)
cap.set(4, height)
classes = ['BinhThuong','Buon','ChanGhet','HanhPhuc','NgacNhien','SoHai','TucGian']

win = Tk()
win.geometry('1380x820')
win.title('NHẬN DIỆN TÂM TRẠNG CỦA BẢN THÂN')
win.resizable(0,0)

win['background']='#000069'
Frame(win,width=1280,height=720,bg='white').place(x=50,y=50)

imagea=Image.open("giaodien.png")
imageb= ImageTk.PhotoImage(imagea)
label1 = Label(image=imageb,
               border=0,
               justify=CENTER)
label1.place(x=50, y=50)

def emotion_face():
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while(True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        frame= cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
                                            gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE
                                        )
        for (x, y, w, h) in  faces:
            try:
                img_test = frame[...,::-1]
                img_test = img_test[y-50:y+h+50, x-50:x+w+50]
                img_test = cv2.resize(img_test,(150,150),interpolation=cv2.INTER_AREA)
                img_test = img_test.astype('float')/255.0
                img_test = img_to_array(img_test)
                img_test = np.expand_dims(img_test,axis=0)
                preds = emotion_model.predict(img_test)
                # print("/nprediction = ",preds)
                label=classes[preds.argmax()]
                # print("/nprediction max = ",preds.argmax())
                # print("/nlabel = ",label)
                label_position = (x,y)
                b = classes[0],":","%.2f" %np.array(preds[0][preds.argmax()]*100),"%"
                cv2.rectangle(frame, (x, y), (x+w, y+h+10), (0, 255, 100), 2)
                cv2.putText(frame,label+b[2]+b[3],label_position,cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            except:
                pass
        cv2.imshow('NHAN DIEN REALTIME', frame)
        c = cv2.waitKey(1)
        if c == 27:
                break
    cap.release() 
    cv2.destroyAllWindows()
def onOpen_video():
    file = askopenfilename(filetypes=(("Video files", "*.mp4;*.flv;*.avi;*.mkv"),
                                       ("All files", "*.*") ))
    if file is not None:
        video_path = str(file)
        cap2 = cv2.VideoCapture(video_path)    
    # print(video_path)
    def emotion_face_video():
        while(True):
            ret, videocap = cap2.read()
            gray = cv2.cvtColor(videocap, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                                            gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE
                                        )
            
            for (x, y, w, h) in  faces:
                try:
                    img_test = videocap[...,::-1]
                    img_test = img_test[y-50:y+h+50, x-50:x+w+50]
                    img_test = cv2.resize(img_test,(150,150),interpolation=cv2.INTER_AREA)
                    img_test = img_test.astype('float')/255.0
                    img_test = img_to_array(img_test)
                    img_test = np.expand_dims(img_test,axis=0)
                    preds = emotion_model.predict(img_test)
                    # print("/nprediction = ",preds)
                    label=classes[preds.argmax()]
                    # print("/nprediction max = ",preds.argmax())
                    # print("/nlabel = ",label)
                    label_position = (x,y)
                    b = classes[0],":","%.2f" %np.array(preds[0][preds.argmax()]*100),"%"
                    cv2.rectangle(videocap, (x, y), (x+w, y+h+10), (0, 255, 100), 2)
                    cv2.putText(videocap,label+b[2]+b[3],label_position,cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                except:
                    pass
            cv2.imshow('NHAN DIEN QUA VIDEO', videocap)
            c = cv2.waitKey(1)
            if c == 27:
                break
        cap2.release() 
        cv2.destroyAllWindows()

    if file is not None:
        emotion_face_video()

def onOpen_img():
    file = askopenfilename(filetypes=(("Images", "*.jpg;*.tif;*.bmp;*.gif;*.png"),
                                       ("All files", "*.*") ))
    if file is not None:
        hinhanh_path = str(file)

    print(hinhanh_path)
    # print(video_path)
    def emotion_face_img():
        img = cv2.imread(hinhanh_path)  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
                                        gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE
                                    )
        
        for (x, y, w, h) in  faces:
            try:
                img_test = img[...,::-1]
                img_test = img_test[y-50:y+h+50, x-50:x+w+50]
                img_test = cv2.resize(img_test,(150,150),interpolation=cv2.INTER_AREA)
                img_test = img_test.astype('float')/255.0
                img_test = img_to_array(img_test)
                img_test = np.expand_dims(img_test,axis=0)
                preds = emotion_model.predict(img_test)
                # print("/nprediction = ",preds)
                label=classes[preds.argmax()]
                # print("/nprediction max = ",preds.argmax())
                # print("/nlabel = ",label)
                label_position = (x,y)
                b = classes[0],":","%.2f" %np.array(preds[0][preds.argmax()]*100),"%"
                cv2.rectangle(img, (x, y), (x+w, y+h+10), (0, 255, 100), 2)
                cv2.putText(img,label+b[2]+b[3],label_position,cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            except:
                pass
        cv2.imshow('NHAN DIEN QUA ANH', img)
        cv2.waitKey(1)
    if file is not None:
        emotion_face_img()

def close():
   win.quit()

def bt_RT(x,y,text,ecolor,lcolor):
    def on_entera(e):
        myButton1['background'] = ecolor #ffcc66
        myButton1['foreground']= lcolor  #000d33

    def on_leavea(e):
        myButton1['background'] = lcolor
        myButton1['foreground']= ecolor

    my_font=('arial', 13, 'bold')
    myButton1 = Button(win,text=text,
                   width=18,
                   height=5,
                   fg=ecolor,
                   border=0,
                   bg=lcolor,
                   font=my_font,
                   activeforeground=lcolor,
                   activebackground=ecolor,
                       command=emotion_face)
                  
    myButton1.bind("<Enter>", on_entera)
    myButton1.bind("<Leave>", on_leavea)

    myButton1.place(x=400,y=520)

bt_RT(400,580,'NHẬN DIỆN REALTIME','white','#000069')

def bt_VD(x,y,text,ecolor,lcolor):
    def on_entera(e):
        myButton1['background'] = ecolor #ffcc66
        myButton1['foreground']= lcolor  #000d33

    def on_leavea(e):
        myButton1['background'] = lcolor
        myButton1['foreground']= ecolor

    my_font=('arial', 13, 'bold')
    myButton1 = Button(win,text=text,
                   width=18,
                   height=5,
                   fg=ecolor,
                   border=0,
                   bg=lcolor,
                   font=my_font,
                   activeforeground=lcolor,
                   activebackground=ecolor,
                    command=onOpen_video)
                  
    myButton1.bind("<Enter>", on_entera)
    myButton1.bind("<Leave>", on_leavea)

    myButton1.place(x=600,y=520)

bt_VD(550, 580,'NHẬN DIỆN TỪ VIDEO','white','#000069')

def bt_ANH(x,y,text,ecolor,lcolor):
    def on_entera(e):
        myButton1['background'] = ecolor #ffcc66
        myButton1['foreground']= lcolor  #000d33

    def on_leavea(e):
        myButton1['background'] = lcolor
        myButton1['foreground']= ecolor

    my_font=('arial', 13, 'bold')
    myButton1 = Button(win,text=text,
                   width=18,
                   height=5,
                   fg=ecolor,
                   border=0,
                   bg=lcolor,
                   font=my_font,
                   activeforeground=lcolor,
                   activebackground=ecolor,
                    command=onOpen_img)
                  
    myButton1.bind("<Enter>", on_entera)
    myButton1.bind("<Leave>", on_leavea)

    myButton1.place(x=800,y=520)

bt_ANH(700,580,'NHẬN DIỆN TỪ ẢNH','white','#000069')

def bt_CL(x,y,text,ecolor,lcolor):
    def on_entera(e):
        myButton1['background'] = ecolor #ffcc66
        myButton1['foreground']= lcolor  #000d33

    def on_leavea(e):
        myButton1['background'] = lcolor
        myButton1['foreground']= ecolor

    my_font=('arial', 13, 'bold')
    myButton1 = Button(win,text=text,
                   width=58,
                   height=3,
                   fg=ecolor,
                   border=0,
                   bg=lcolor,
                   font=my_font,
                   activeforeground=lcolor,
                   activebackground=ecolor,
                    command=close)
                  
    myButton1.bind("<Enter>", on_entera)
    myButton1.bind("<Leave>", on_leavea)

    myButton1.place(x=400,y=645)

bt_CL(400,580,'EXIT','white','#B21F00')

win.mainloop()
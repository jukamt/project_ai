import numpy as np
import cv2
import av
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase,  RTCConfiguration


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

try:
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    classes = ("BinhThuong","Buon","ChanGhet","HanhPhuc","NgacNhien","SoHai","TucGian")
    model=load_model('model/emotion_model.h5')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            try:
                cv2.rectangle(img=img, pt1=(x, y), pt2=(
                                x + w, y + h), color=(0, 255, 100), thickness=2)
                img_gray = img[...,::-1]
                roi_gray = img_gray[y-50:y+h+50, x-50:x+w+50]
                roi_gray = cv2.resize(roi_gray, (150, 150), interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi_test = np.expand_dims(roi, axis=0)
                    preds = model.predict(roi_test)
                    labels = classes[preds.argmax()]
                cv2.putText(img, labels, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except:
                pass
        return av.VideoFrame.from_ndarray(img, format='bgr24')

def main():
    # Face Analysis Application #
    st.title("TERM PROJECT OF COURSE “ARTIFICIAL INTELLIGENCE”")
    activiteis = ["Home", "Nhận diện Realtime", "Thông tin"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Name: Dang Nguyen Minh Tien\n
            Email: 19146401@student.hcmute.edu.vn
            Phone:  +84 569 014 570""")
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Nhan dien tam trang ban than.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 GVHD: PSG.TS Nguyễn Trường Thịnh.\n
                 SVTH: Đặn Nguyễn Minh Tiến.\n
                 MSSV: 19146401.\n
                 Nhóm: 07
                 """)
    elif choice == "Nhận diện Realtime":
        st.header("Nhan dien tam trang ban than Realtime")
        st.write("Click on start to use webcam")
        webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

    elif choice == "Thông tin":
        st.subheader("Thông tin về đề tài")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Nhan dien tam trang ban than.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">Term Project - Course "Artificial Intelligence" HCMUTE - FME. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()
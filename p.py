import cv2,os
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    # ret, frame = video_capture.read()
    frame=img=cv2.imread('fr/u/b.jpg')
    # Resize frame of video to 1/4 size for faster face recognition processing
    

    # Display the resulting image
    # cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()
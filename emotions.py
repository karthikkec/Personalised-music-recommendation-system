process_this_frame = True
ba={
    'Chaarz,Happy':'Nenjil Nenjil.mp3',
'KarthikM,Surprised':'ASirikkalamParakkalamMassTamilanioMusic.mp3',
'karthika,Happy':'03 Chillax.mp3',
'keerthanaD,Happy':'DARBARTamilChummaKizhiLyricVideoRajinikanthARMurugadossRingtone.mp3',
'Dr.K.Kousalya,Happy':'Singapenne.mp3',
}
def fv(frame=None):
    global last,process_this_frame
    # if  frame and not frame.any():
    img=frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    name='Unknown'
    # Only process every other frame of video to save time
    if process_this_frame or 1:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        if maxindex!=last and 0:
            if maxindex==3:
                threading.Thread(play('03 Chillax.mp3'))
            elif maxindex==5:
                threading.Thread(play('06.Nenjil Nenjil.mp3'))
            else:
                threading.Thread(play('06.Nenjil Nenjil.mp3'))
        last=maxindex
        # threading.Thread(play(ba.get(f'{name.split(".")[0]},{emotion_dict[maxindex]}','03 Chillax.mp3')))
        

        print(name,emotion_dict[maxindex])
        sleep(5)
        # cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    
for file in os.listdir('fr/u/'):
    print(file)
    fv(cv2.imread(f'fr/u/{file}'))   

  
# emotions will be displayed on your face from the webcam feed
if mode == "display" or 0:
    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        fv(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
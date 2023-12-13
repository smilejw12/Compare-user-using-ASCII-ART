def clear_database():
    # 사용자 데이터를 모두 삭제하는 함수
    with open('face_encodings.pickle', 'wb') as f:
        pickle.dump([], f, pickle.HIGHEST_PROTOCOL)

    with open('face_names.pickle', 'wb') as f:
        pickle.dump([], f, pickle.HIGHEST_PROTOCOL)
    
    with open('passwords.pickle', 'wb') as f:
        pickle.dump([], f, pickle.HIGHEST_PROTOCOL)


import face_recognition
import cv2
import numpy as np
import secrets
import string
import pickle
import pyminizip


# 메뉴 출력
print("┌────────────────────────────────────────────── 파일 암호화  메뉴 ─────────────────────────────────────────────────┐\n")
print("│                                                                                                                  │\n")
print("│                                                                                                                  │\n")
print("│   1. 파일 암호화                                                                                                 │\n")
print("│                                                                                                                  │\n")
print("│                                                                                                                  │\n")
print("│   2. 파일 암호화 해제                                                                                            │\n")
print("│                                                                                                                  │\n")
print("│                                                                                                                  │\n")
print("│   3. 저장된 이용자 삭제(주의! 기존에 암호화된 파일 해제 불가)                                                       │\n")
print("│                                                                                                                  │\n")
print("│                                                                                                                  │\n")
print("└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")



video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

known_face_encodings = []
known_face_names = []
passwords = []

with open('face_encodings.pickle', 'rb') as f:
    known_face_encodings = pickle.load(f)

with open('face_names.pickle', 'rb') as f:
    known_face_names = pickle.load(f)

with open('passwords.pickle', 'rb') as f:
    passwords = pickle.load(f)


# 사용자의 얼굴을 등록하거나 암호화된 파일을 열기 위한 옵션 선택
option = input("\n\n옵션을 선택해주세요: ")

if option == "1":
    #사용자 등록 후 파일 암호화
    print("\x1B[H\x1B[J")
    username = input("사용자의 이름을 알려주세요\n")
    face_locations = []
    face_encodings = [] 
    face_names = []
    process_this_frame = True


    registered = False
    print("\x1B[H\x1B[J")
    print("얼굴을 인식하고 있습니다...")
   
   #사용자 얼굴이 인식되면 종료
    while not registered:
        ret, frame = video_capture.read()

        # 속도를 높이기 위해서 사용자 얼굴의 사이즈를 0.25배 
        small_frame = cv2.resize(frame, (0, 0), fx=.25, fy=.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 시간절약을 위해 비디오의 프레임을 1만 처리
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                              
                #Pickle 파일 업데이트
                known_face_encodings.append(face_encoding)
                known_face_names.append(username)

                with open('face_encodings.pickle', 'wb') as f:
                    pickle.dump(known_face_encodings, f, pickle.HIGHEST_PROTOCOL)

                with open('face_names.pickle', 'wb') as f:
                    pickle.dump(known_face_names, f, pickle.HIGHEST_PROTOCOL)


                #얼굴이 발견되면 시스템에 이름 추가
                #그런 다음에 캠 종료
                registered = True


                #현재 얼굴이 저장된 사용자와 일치하는지 테스트한다
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        #얼굴 인식이 가능하면 얼굴에 사각형 프레임을 띄운다
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        # 얼굴 인식이 오랫동안 안되면 q를 누르면 정지된다
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    
       
    print("사용자 인식이 완료됐습니다!\n")
    src_files = input("어서오세요! " + username + ". 암호화할 파일을 선택하세요:\n").split(" ")
    archive_name = input("\n암호화될 파일명을 입력하세요(영문만 가능합니다):\n")

    src_paths = []

    for files in src_files:
        src_paths.append(archive_name)
    
    alphabet = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(secrets.choice(alphabet) for i in range(2000))
    passwords.append(password)
    with open('passwords.pickle', 'wb') as f:
        pickle.dump(passwords, f, pickle.HIGHEST_PROTOCOL) 

    print("압축중...")
    
    #암호화 압축
    compression_level = 5 # 1-9
    pyminizip.compress_multiple(src_files, src_paths, ".\\" + archive_name + ".zip", password, compression_level)

    print("\n압축과 암호화가 완료됐습니다!")



elif option == "2":
    face_locations = []
    face_encodings = [] 
    face_names = []
    process_this_frame = True

    current_person = ""
    current_password = ""
    userfound = False
    print("얼굴을 인식중입니다...")

    while not userfound:
        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                
                
                #will take the first face it sees, can restrict it to only allow 1 later
                #store in file of registered users

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
                    current_person = name

                    current_password = passwords[best_match_index]
                    userfound = True

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

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


    target = input("어서오세요 " + current_person +"!. 암호를 해독할 파일을 골라주세요:\n")
    print("해독중...")
    pyminizip.uncompress(target, current_password, None, 0)
    print("암호 해독이 완료됐습니다!")
    
elif option == "3":
    clear_database()
    print("\x1B[H\x1B[J")
    print("---모든 사용자 삭제 완료---")
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os


# Creating a pose instance
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(landmark1, landmark2, landmark3,select=''):
    if select == '1':
        x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
        x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
        x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

        angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    else:
        radians = np.arctan2(landmark3[1] - landmark2[1], landmark3[0] - landmark2[0]) - np.arctan2(landmark1[1] - landmark2[1], landmark1[0] - landmark2[0])
        angle = np.abs(np.degrees(radians))
    
    angle_calc = angle + 360 if angle < 0 else angle
    return angle_calc

def correct_feedback(model,video='0',input_csv='0'):
    # Load video
    cap = cv2.VideoCapture(video)  # Replace with your video path

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    
    accurate_angle_lists = []
    

    angle_name_list = ["L-wrist","R-wrist","L-elbow", "R-elbow","L-shoulder", "R-shoulder", "L-knee", "R-knee","L-ankle","R-ankle","L-hip", "R-hip"]
    angle_coordinates = [[13, 15, 19], [14, 16, 18], [11, 13, 15], [12, 14, 16], [13, 11, 23], [14, 12, 24], [23, 25, 27], [24, 26, 28],[23,27,31],[24,28,32],[24,23,25],[23,24,26]]
    correction_value = 30

    fps_time = 0
   
    while cap.isOpened():
        ret_val, image = cap.read()
        
        if not ret_val:
            break

        # Convert the image to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize_rgb = cv2.resize(image_rgb, (0, 0), None, .50, .50)
        # Get the pose landmarks
        results = pose.process(image_rgb)
        #save angle main
        angles = []
        
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            # Get the angle between the left elbow, wrist and left index points.
            left_wrist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value],'1')
            angles.append(left_wrist_angle)
            # Get the angle between the right elbow, wrist and left index points.
            right_wrist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value],'1')
            angles.append(right_wrist_angle)


            # Get the angle between the left shoulder, elbow and wrist points.
            left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],'1')
            angles.append(left_elbow_angle)
            # Get the angle between the right shoulder, elbow and wrist points.
            right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],'1')
            angles.append(right_elbow_angle)
            # Get the angle between the left elbow, shoulder and hip points.
            left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],'1')
            angles.append(left_shoulder_angle)

            # Get the angle between the right hip, shoulder and elbow points.
            right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],'1')
            angles.append(right_shoulder_angle)

            # Get the angle between the left hip, knee and ankle points.
            left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],'1')
            angles.append(left_knee_angle)

            # Get the angle between the right hip, knee and ankle points
            right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],'1')
            angles.append(right_knee_angle)

            # Get the angle between the left hip, ankle and LEFT_FOOT_INDEX points.
            left_ankle_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value],'1')
            angles.append(left_ankle_angle)

            # Get the angle between the right hip, ankle and RIGHT_FOOT_INDEX points
            right_ankle_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value],'1')
            angles.append(right_ankle_angle)

            # Get the angle between the left knee, hip and right hip points.
            left_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],'1')
            angles.append(left_hip_angle)

            # Get the angle between the left hip, right hip and right kneee points
            right_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],'1')
            angles.append(right_hip_angle)
            
            y = model.predict([angles])
            
            Name_Yoga_Classification = str(y[0])

            probabilities = model.predict_proba([angles])
            print([probabilities])
            class_labels = model.classes_
            check_accry_class = False
            for i,class_label in enumerate(class_labels):
                probability = probabilities[0, i]
                if probability > 0.5 :
                    check_accry_class = True
                else:
                    continue

            with open(input_csv, 'r') as inputCSV:
                for row in csv.reader(inputCSV):
                    if row[12] == Name_Yoga_Classification: 
                        accurate_angle_lists = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11])]
                
            folder_path = 'F:/Anaconda_Project/Utilizing_Deep_Learning_for_Human_Pose_Estimation_in_Yoga/teacher_yoga/angle_teacher_yoga.csv'

            prefix_to_match = Name_Yoga_Classification


            if check_accry_class == True :
              
                (w, h), _ = cv2.getTextSize(Name_Yoga_Classification, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(image, (10, image.shape[0] - 30), (10 + w, image.shape[0] - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, Name_Yoga_Classification, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            else :
                # Display the classification result in the bottom-left corner
                (w, h), _ = cv2.getTextSize('None', cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(image, (10, image.shape[0] - 30), (10 + w, image.shape[0] - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, 'None', (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            
            
            correct_angle_count = 0
            for itr in range(12):
                point_a = (int(landmarks[angle_coordinates[itr][0]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][0]].y * image.shape[0]))
                point_b = (int(landmarks[angle_coordinates[itr][1]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][1]].y * image.shape[0]))
                point_c = (int(landmarks[angle_coordinates[itr][2]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][2]].y * image.shape[0]))

                angle_obtained = calculate_angle(point_a, point_b, point_c,'0')
                    
                if angle_obtained < accurate_angle_lists[itr] - correction_value:
                    status = "more"
                elif accurate_angle_lists[itr] + correction_value < angle_obtained:
                    status = "less"
                else:
                    status = "OK"
                    correct_angle_count += 1

                # Display status
                status_position = (point_b[0] - int(image.shape[1] * 0.03), point_b[1] + int(image.shape[0] * 0.03))
                cv2.putText(image, f"{status}", status_position, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


                cv2.putText(image, f"{angle_name_list[itr]}", (point_b[0] - 50, point_b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)                  
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            
            posture = "CORRECT" if correct_angle_count > 9 else "WRONG"
            posture_color = (0, 255, 0) if posture == "CORRECT" else (0, 0, 255)  

            posture_position = (10, 30)  
            cv2.putText(image, f"Yoga movements: {posture}", posture_position, cv2.FONT_HERSHEY_PLAIN, 1.5, posture_color, 2)

            fps_text = f"FPS: {1.0 / (time.time() - fps_time):.3f}" 
            cv2.putText(image, fps_text, fps_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            

            cv2.imshow('Mediapipe Pose Estimation', image)
            fps_time = time.time()



            

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

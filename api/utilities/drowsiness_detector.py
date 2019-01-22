from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import socket
# from morpheus1.api import views
import json


# create socket connection
HOST = "192.168.43.183"
# HOST = socket.gethostname()
# HOST = "Aditya"
print(HOST)
PORT = 4321

s = socket.socket()
s.bind((HOST, PORT))
s.listen(1)
c, addr = s.accept()
print("Connection from: ", str(addr), "established!")


def eye_aspect_ratio(eye):
    """
    This function calculates the Eye Aspect Ratio in real time on a frame
    by frame basis for each eye
    :param eye: the landmark coordinates of an eye
    :return: Eye Aspect Ratio (EAR)
    """
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor dataset")
args = vars(ap.parse_args())


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below te threshold for to set off the alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 5

# initialize the frame counter as well as a boolean used to
# indicate if the alarm should go off
COUNTER = 0
ALARM_ON = "False"

# initialise the dlib face detector (HOG-linear SVM object detector) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()  # detects faces
predictor = dlib.shape_predictor(args["shape_predictor"])  # finds eye position

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")

# loop over frames from the video stream
while True:
    # grab the frame from the video stream, resize it, and convert
    # to grayscale channels
    # imgResp = views.DataUpdateView.post("send")
    message = "Hi"
    c.send(message.encode('utf-8'))
    imgResp = c.recv(1024)  # max size = 2^17, dtype = byte array
    print("hi ", type(imgResp))
    imgNp = np.array(bytearray(imgResp), dtype=np.uint8)  # converting byte array to numpy
    frame = cv2.imdecode(imgNp, -1)
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmarks (x, y)-coordinates to a numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # if the eyes were closed for a sufficient number of frames
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if ALARM_ON == "False":
                    ALARM_ON = "True"
                    print("Alarm sounded !!!")

                # draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = "False"

        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame and send data to app
        # message = str(ear) + "," + ALARM_ON
        message = json.dumps({'EAR': ear, "ALARM": ALARM_ON})  # converting to JSON
        """
        Over here I need to post MESSAGE back to Aditya's app using Django by converting it into a JSON
        """
        c.send(message.encode('utf-8'))
        cv2.imshow("Video Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break from the loop
        if key == ord("q"):
            break


# cleaning up
cv2.destroyAllWindows()
c.close()

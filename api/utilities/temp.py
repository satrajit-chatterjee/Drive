import base64
import json

message = bytes(4)  # this is the exact image type
w = base64.encodebytes(message).decode('ascii')  # Aditya will send this exact format to me
r = {'is_claimed': 'True', 'rating': w}
r = json.dumps(r)
loaded_r = json.loads(r)
# loaded_r['rating']  # Output 3.5
# print(type(r))  # Output str
print(loaded_r)  # Output dict

# w = base64.encodebytes(message).decode('ascii')
# print(w)
print(base64.b64decode(""))

# import os
#
# print(os.open('shape_predictor_68_face_landmarks.dat', flags=1))

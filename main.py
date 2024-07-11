from pynput import keyboard, mouse
from deepface import DeepFace
import cv2
import json
import numpy as np
from ctypes import CDLL
import time

loginPF = CDLL('/System/Library/PrivateFrameworks/login.framework/Versions/Current/login')

def lock_screen():
    loginPF.SACLockScreenImmediate()

current_frame = None
current_frame_verified = False
trained_face_embedding = None
consecutive_failures = 0
max_consecutive_failures = 3
base_threshold = 1.17 # from https://github.com/serengil/deepface/blob/master/deepface/modules/verification.py

def euclidean_distance(embedding1, embedding2):
    if isinstance(embedding1, list):
        embedding1 = np.array(embedding1)
    if isinstance(embedding2, list):
        embedding2 = np.array(embedding2)
    return np.linalg.norm(embedding1 - embedding2)

def l2_normalize(x):
    if isinstance(x, list):
        x = np.array(x)
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def find_distance(embedding1, embedding2):
    if isinstance(embedding1, list):
        embedding1 = np.array(embedding1)
    if isinstance(embedding2, list):
        embedding2 = np.array(embedding2)
    
    # return euclidean_distance(l2_normalize(embedding1), l2_normalize(embedding2))
    return euclidean_distance(embedding1, embedding2)
    # cosine distance
    # a = np.matmul(np.transpose(embedding1), embedding2)
    # b = np.sum(np.multiply(embedding1, embedding1))
    # c = np.sum(np.multiply(embedding2, embedding2))
    # return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def verify():
    global current_frame
    global trained_face_embedding
    global base_threshold
    global current_frame_verified

    if current_frame is None:
        return None
    
    result = False
    faces = DeepFace.extract_faces(frame, detector_backend="opencv", align=True, anti_spoofing=True, enforce_detection=False)

    for face in faces:
        if not face['is_real']:
            print(f"Found spoofed face, antispoof score: {face['antispoof_score']}")
            # save file with current timestamp
            cv2.imwrite(f'spoofed-{time.time()}.jpg', frame)
            break

        emb = DeepFace.represent(face["face"], model_name='VGG-Face', detector_backend="skip", align=True, normalization="VGGFace")[0]
        # print('Face detected with confidence:', face["confidence"])
        
        min_distance = 1e9
        for trained_emb in trained_face_embedding:
            distance = find_distance(trained_emb["embedding"], emb["embedding"])
            if distance < min_distance:
                min_distance = distance
            if distance <= base_threshold:
                result = True
                break
        if not result:
            print('Mismatching face. Distance:', min_distance)
    current_frame_verified = True
    return result


def on_press(key):
    global consecutive_failures
    if not current_frame_verified:
        result = verify()
        if result is None:
            return
        if not result:
            consecutive_failures += 1
            print(f'Keypress: {key}, Consecutive failures: {consecutive_failures}')
        else:
            consecutive_failures = 0
            # print('Match - resetting consecutive failures')
    if consecutive_failures >= max_consecutive_failures:
        # save the current frame
        cv2.imwrite(f'intruder-{time.time()}.jpg', current_frame)
        print('Locking screen...')
        consecutive_failures = 0
        lock_screen()

if __name__ == '__main__':
    with open('face.json') as f:
        trained_face_embedding = json.load(f)
        print(f'Found {len(trained_face_embedding)} trained embeddings')

    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    mouse_listener = mouse.Listener(
        on_click=lambda x, y, button, pressed: on_press(f'mouse click {x}, {y}, {button}, {pressed}'),   
    )
    mouse_listener.start()

    vid = cv2.VideoCapture(0)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print('FPS:', fps)
    max_confidence = 0
    i = 0

    while(True): 
        ret, frame = vid.read()
        i += 1
        if i % 5 != 0:
            continue
        frame = cv2.flip(frame,1)
        current_frame = frame
        current_frame_verified = False
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    vid.release()
    cv2.destroyAllWindows()
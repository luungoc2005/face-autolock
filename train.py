from deepface import DeepFace
import cv2
import json
from os import path
import numpy as np

base_threshold = 1.17

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

def verify(emb, embs):
    for e in embs:
        distance = find_distance(emb["embedding"], e["embedding"])
        if distance < base_threshold:
            return True
    return False

if __name__ == '__main__':
    vid = cv2.VideoCapture(0)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print('FPS:', fps)
    confidence_threshold = .94
    i = 0

    existing_embs = []
    if path.isfile('face.json'):
        with open('face.json') as f:
            existing_embs = json.load(f)

    while(True):
        ret, frame = vid.read()
        i += 1
        if i % 5 != 0:
            continue
        frame = cv2.flip(frame,1)
        try:
            # embedding = DeepFace.represent(frame, model_name='DeepID', enforce_detection=False)
            faces = DeepFace.extract_faces(frame, detector_backend="opencv", align=True, grayscale=False, anti_spoofing=False, enforce_detection=False)
            
            if len(faces) == 1:
                embedding = DeepFace.represent(faces[0]["face"], model_name='VGG-Face', detector_backend="skip", normalization="VGGFace", align=True)
                
                if len(embedding) == 1 and faces[0]["confidence"] >= confidence_threshold:
                    current_emb = embedding[0]
                    if not verify(current_emb, existing_embs):
                        print('New face detected with confidence:', confidence_threshold)
                        existing_embs.append(current_emb)
                        with open('face.json', 'w') as f:
                            json.dump(existing_embs, f, indent=2)
        except:
            pass
        
        cv2.imshow('webcam', frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    vid.release()
    cv2.destroyAllWindows()
from deepface import DeepFace
import cv2
import json

if __name__ == '__main__':
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
        try:
            # embedding = DeepFace.represent(frame, model_name='DeepID', enforce_detection=False)
            faces = DeepFace.extract_faces(frame, detector_backend="opencv", align=True, grayscale=False, anti_spoofing=False, enforce_detection=False)
            
            if len(faces) == 1:
                embedding = DeepFace.represent(faces[0]["face"], model_name='VGG-Face', detector_backend="skip", normalization="VGGFace", align=True)
                
                if len(embedding) == 1 and faces[0]["confidence"] > max_confidence:
                    print('Face detected with confidence:', max_confidence)
                    with open('face.json', 'w') as f:
                        json.dump(embedding, f, indent=2)
                        print('Face saved to face.json')
                    max_confidence = faces[0]["confidence"]
        except:
            pass
        
        cv2.imshow('webcam', frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    vid.release()
    cv2.destroyAllWindows()
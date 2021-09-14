import cv2
import numpy as np
from tensorflow.keras.models import load_model
model=load_model("./facemask_detection_model")
results={0:'Mask',1:'No Mask'}
# Mask is green square, No Mask is red square
color={0:(0,255,0),1:(0,0,255)}

square_size = 4 # Rescale factor for face image size
cap = cv2.VideoCapture(0) # Accessing computer's camera

# Using Haarcascade for frontal face detection, would need to change this to the correct directory for classifier
haarcascade = cv2.CascadeClassifier('/Users/User/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
counter = 0 # Counter for frames with mask on
while True:

    # grabing image array vector of image from camera
    ret, frame = cap.read()

    frame=cv2.flip(frame,1,1) # Flip camera so it is no longer mirror-reversal of original 
    
    # Setting expected size of face image to be captured from camera for detection
    smaller_frame = cv2.resize(frame, (frame.shape[1] // square_size, frame.shape[0] // square_size))
    faces = haarcascade.detectMultiScale(smaller_frame)
    for (x, y, w, h) in faces:
        # rescale x, y, width, height by square size
        x = x * square_size
        y = y * square_size
        w = w * square_size
        h = h * square_size
        
        
        face_img = frame[y:y+h, x:x+w] # Setting image array vector size

        # Reshaping and resizing image vector to fit trained model
        adjusted_frame_size=cv2.resize(face_img,(224,224)) 
        prediction_img=np.reshape(adjusted_frame_size,(1,224,224,3))
    
        # Predicting captured image using trained model 
        prediction=model.predict(prediction_img)
        
        # Extracting the most likely outcome
        result=np.argmax(prediction[0])

        probability = np.amax(prediction[0])
      
        cv2.rectangle(frame,(x,y),(x+w,y+h),color[result],1) # Create rectangle frame that highlights prediction
        cv2.rectangle(frame,(x,y-40),(x+w,y),color[result],-1) # Rectangle for text 
        cv2.putText(frame, results[result] + ' ' + str(round((probability * 100), 2)) + '%', (x+40, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2) # Text to display prediction
        if result == 0 and probability > 0.99: # At least 99% sure Mask is on
            counter += 1
        else:
            counter = 0
        if counter >= 90: # 90 frames in a row with mask on will confirm mask is on/pass the detection test
            cv2.putText(frame, "User has a mask on", (x-10, y-120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        elif counter >= 25:
            cv2.putText(frame, "Stand still with mask on", (x-10, y-120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
        else:
            cv2.putText(frame, "Please put on your mask", (x-20, y-120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
    cv2.imshow('FaceMaskDetection', frame) # Read image from captured frame
    if cv2.waitKey(1) == 27: # esc key to exit program
        break
cap.release()
cv2.destroyAllWindows()
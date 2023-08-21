import numpy as np
import cv2
import copy
import os
import math
from make_video import make_video
from progress.bar import Bar


def main():
    capture = cv2.VideoCapture('HD Security.mp4')
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=500)
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    try:
        # creating a folder named data 
    	if not os.path.exists('frames'): 
    		os.makedirs('frames') 
    except OSError: 
    	print ('Error: Creating directory of data') 
    
    bar = Bar('Processing Frames', max=length)


    
    first_iteration_indicator = 1
    # for i in range(0, length):
    for i in range(0, 600):

        ret, frame = capture.read()
        
        
        # If first frame
        if first_iteration_indicator == 1:

            first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
        else:

            filter = background_subtractor.apply(frame)  # remove the background
            
            # make black frame
            ret, filter = cv2.threshold(filter, 0, 0, cv2.THRESH_BINARY)

            
            # make greyscale for faster detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # detect people -> boxes
            # returns the bounding boxes for the detected objects
            boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05, hitThreshold=0)

            boxes = np.array([[math.ceil(x + w/8*3), math.ceil(y + h/8*3), math.ceil(x + w/8*5), math.ceil(y + h/8*5)] for (x, y, w, h) in boxes])
            # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
            
            for (xA, yA, xB, yB) in boxes:
                # display the detected boxes in the colour picture
                cv2.rectangle(filter, (xA, yA), (xB, yB), (255, 255, 255), -1)
            
            threshold = 2
            maxValue = 2
            ret, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)
            
            # detect faces -> faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
            # Apply a Gaussian blur to faces
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                
                # Apply Gaussian blur kernel size of (51, 51)
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                frame[y:y+h, x:x+w] = blurred_roi
            
            cv2.imwrite('./frame.jpg', frame)
            cv2.imwrite('./diff-bkgnd-frame.jpg', filter) 
            
            # add accumulated image
            accum_image = cv2.add(accum_image, th1)
            cv2.imwrite('./mask.jpg', accum_image)
            
            
            # normal with red heat start
            color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
            video_frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.7, 0)

            name = "./frames/frame%d.jpg" % i
            cv2.imwrite(name, video_frame)
            # normal with red heat end
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        bar.next()

    bar.finish()

    make_video('./frames/', './output.avi')

    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

    # save the final heatmap
    cv2.imwrite('diff-overlay.jpg', result_overlay)

    # cleanup
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

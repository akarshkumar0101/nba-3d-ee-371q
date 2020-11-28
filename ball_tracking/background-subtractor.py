import cv2 as cv

DIR_PATH = 'nba-3d-data/harden/view_0/frame_%05d.png'
VIDEO_PATH = 'nba-3d-data/harden/all_views.mp4'
# background subtraction on all [files], isolate out the players (hopefully) in a single view
backSub = cv.createBackgroundSubtractorMOG2()
capture = cv.VideoCapture(cv.samples.findFileOrKeep(DIR_PATH))
while(True):
    ret,frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)

    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey()
    if keyboard == 'q' or keyboard == 27:
        break
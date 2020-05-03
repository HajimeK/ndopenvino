import cv2

resource = "/home/hajime/git/nd131-openvino-fundamentals-project-starter/resources/Pedestrian_Detect_2_1_1.mp4"
resource = 0

capture = cv2.VideoCapture(resource)

while(True):
    ret, frame = capture.read()
    # resize the window
    windowsize = (800, 600)
    frame = cv2.resize(frame, windowsize)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('title',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
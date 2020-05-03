import argparse
import cv2
import numpy as np

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Handle an input stream")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc)
    args = parser.parse_args()

    return args


def capture_stream(args):

    ### TODO: Handle image, video or webcam
    capture = cv2.VideoCapture(args.i if args.i != None else 0)
    #capture = cv2.VideoCapture(0)

    while(True):
        ### TODO: Get and open video capture
        ret, frame = capture.read()
        if ret:
            ### TODO: Re-size the frame to 100x100
            windowsize = (800, 600)
            frame = cv2.resize(frame, windowsize)

            ### TODO: Add Canny Edge Detection to the frame, 
            ###       with min & max values of 100 and 200
            ###       Make sure to use np.dstack after to make a 3-channel image
            edges = cv2.Canny(frame,100,200)

            ### TODO: Write out the frame, depending on image or video
            cv2.imshow("diaplay", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ### TODO: Close the stream and any windows at the end of the application
    capture.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    capture_stream(args)


if __name__ == "__main__":
    main()

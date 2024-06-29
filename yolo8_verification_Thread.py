"""
    Ultralytics yolo8 verification thread.
    Basically an intermediate queue is created and filled with zoomed detections
    from the TPU, CPU, etc. AI and then yolo8 is run on the zoomed in
    image and if person is detected the image is sent to the results queue.
    
    On 19-8750H desktop with Nvidia RTX3070:
    python3 AI.py -nTPU 1 -y8v -d 1 -cam 6onvif.txt -rtsp 19cams.rtsp
    Yielded ~75 fps with 25 cameras for an ~80660 second test run.
    There were 6,064,648 frames processed by the TPU with 3595 persons detected.
    The yolo8 verification accepted 3024 and rejected 571.
    
    My review suggests almost all the rejections were false negatives, but a fair price to pay
    for the near complete rejection of false positives from my bogus detection collection.
"""

from ultralytics import YOLO
import datetime
from imutils.video import FPS
import cv2


global __Thread__
__Thread__ = False

global __verifyConf__
__verifyConf__ = 0.65

global __y8modelSTR__
__y8modelSTR__ = 'yolo8/yolov8x.pt'

global model

'''
one time code to run when thread is launched.
'''
def threadInit():
    global model
    global __y8modelSTR__
    # Load the YOLOv8 model
    model = YOLO(__y8modelSTR__)
    print("[INFO] Using " + __y8modelSTR__ + " yolo8 pre-trained model")
    return



def do_inference( image, model,  confidence ):
    
    boxPoints=(0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength
    personDetected = False
    detectConfidence = 0.0
    # call code to do an inference
    results = model.predict(image, conf=confidence-0.001, verbose=False)
    # Visualize the results on the image
    annotated_image = results[0].plot(line_width=1, labels=False)
    for result in results:
        boxes=result.boxes
        for i in range(len(boxes.data)):
            if int(boxes.data[i][5].item()) == 0 and boxes.data[i][4].item() > confidence:
                personDetected = True
                detectConfidence = boxes.data[i][4].item()
                startX = int(boxes.data[i][0].item())
                startY = int(boxes.data[i][1].item())
                endX = int(boxes.data[i][2].item())
                endY = int(boxes.data[i][3].item())
                xlen=endX-startX
                ylen=endY-startY
                xcen=int((startX+endX)/2)
                ycen=int((startY+endY)/2)
                boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                break
    #print(type(annotated_image), annotated_image.shape)
    return annotated_image.copy(), personDetected, boxPoints, detectConfidence



def yolov8_thread(results, yoloQ):
    global __Thread__
    global __verifyConf__

    print("Starting Yolo v8 verification thread...")
    if yoloQ is None:
        print(    "ERROR! no yolo Queue!")

    threadInit()
    yoloVerified=0
    yoloRejected=0
    yoloWaited = 0
    dcnt=0
    ecnt=0
    ncnt=0
    print("Yolo v8 verification thread is running...")
    __Thread__ = True

    while __Thread__ is True:
        try:
            # ssd_frame is full camera resolution with SSD detection box overlaid
            # yolo_frame is "zoomed in" on the SSD detection box and resized to 608x608 for darknet yolo4 inference
            #yoloQ.put((image, cam, personDetected, imageDT, aiStr, boxPoints, yolo_frame), True, 1.0)
            ssd_frame, cam, personDetected, imageDT, ai, boxPoints, yolo_frame = yoloQ.get(True, 1.0)
        except:
            yoloWaited+=1
            continue

        try:
            # Note these boxpoints are for the persons in the verification image which we ignore here.
            image, personDetected, _, detectConfidence = do_inference( yolo_frame, model, __verifyConf__ )
            if personDetected is True:   # yolov4 has verified the MobilenetSSDv2 person detection
                ## image is the yolo_frame with the yolo detection boxes overlaid (from the boxpoints).
                ## boxPoints are from the SSD inference.
                # draw the verification confidence onto the ssd_frame
                yoloVerified+=1
                text = "Yolo8: {:.1f}%".format(detectConfidence * 100)   # show verification confidence on detection image
                cv2.putText(ssd_frame, text, (2, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                if results.full():
                    [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                    dcnt+=1                       
                results.put((ssd_frame, cam, True, imageDT, ai, boxPoints, image.copy()), True, 1.0)
                ###print(detections, boxpoints)    # lets take a look at what we are getting
            else:
                yoloRejected+=1
                if results.full():
                    [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                    ncnt+=1                       
                results.put((ssd_frame, cam, False, imageDT, ai, (-2,0, 0,0, 0,0, 0,0), image.copy()), True, 1.0)
        except Exception as e:
            ecnt+=1
            print('[Exception] yolo_thread'+ str(cam) + ': ' + str(e))
            continue
    print("Yolo v8 frames Verified: {}, Rejected: {},  Waited: {} seconds.".format(str(yoloVerified), str(yoloRejected), str(yoloWaited)))
    print("    Verified dropped: " + str(dcnt) + " results dropped: " + str(ncnt) + " results.put() exceptions: " + str(ecnt))



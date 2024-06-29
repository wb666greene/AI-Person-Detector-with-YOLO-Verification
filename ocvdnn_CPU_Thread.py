#! /usr/bin/python3
'''
    This is becoming unweildy to maintain and getting more difficult to test.
    It basically lets CPU only AI run on a system without AI co-processor or OpenVINO.
    With a "weak" Nvidia GPU yolo verification may improve the high false
    positive rate of MobilenetSSD_v1 So I am attempting to add yolo verification.

    If this proves too difficult I'll just remove the ocvdnn support.
'''

import numpy as np
import cv2
import datetime
from PIL import Image
from io import BytesIO
from imutils.video import FPS


global __Thread__
__Thread__ = False

global __VERIY_DIMS__
__VERIFY_DIMS__ = (300,300)

# *** OOpenCV DNN CPU AI Thread ***
#******************************************************************************************************************
#******************************************************************************************************************
# function for CPU AI detection, trivial change from MYRIAD version
def AI_thread(results, inframe, net, tnum, cameraLock, nextCamera, Ncameras,
                PREPROCESS_DIMS, confidence, verifyConf, dnnStr, blobThreshold, yoloQ):
    global __Thread__
    global __VERIY_DIMS__

    print("[INFO] openCV dnn module CPU AI thread" + str(tnum) + " is running...")
    if yoloQ is not None:
        print("    openCV dnn module thread" + str(tnum) + " is using yolo verification.")
    __Thread__ = True
    waits=0
    dcnt=0
    ncnt=0
    ecnt=0
    fcnt=0
    detect=0
    noDetect=0
    personIdx=15
    DNN_verify_fail=0
    prevDetections=list()
    for i in range(Ncameras):
        prevDetections.append(0)
    if tnum > 0:
        dnnTarget = dnnStr + str(tnum)
    else:
        dnnTarget = dnnStr
    aiStr = dnnTarget
    cfps = FPS().start()
    while __Thread__ is True:
     # I'm not sure later versions fix it, or if its confined to the Raspberry Pi camera module, but I've left it in here.
        cameraLock.acquire()
        cq=nextCamera
        nextCamera = (nextCamera+1)%Ncameras
        cameraLock.release()
        # get a frame
        try:
            (image, cam, imageDT) = inframe[cq].get(True,0.100)
        except:
            image = None
            waits+=1
            continue
        if image is None:
            continue
        (H, W) = image.shape[:2]
        orig_image=image.copy()   # for zoomed in verification run
        blob = cv2.dnn.blobFromImage(cv2.resize(image, PREPROCESS_DIMS), 0.007843, PREPROCESS_DIMS, 127.5)
        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()
        cfps.update()    # update the FPS counter
        fcnt+=1
        # loop over the detections, pretty much straight from the PyImageSearch sample code.
        personDetected = False
        DNNdetect=False
        boxPoints=(0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength
        for i in np.arange(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]   # extract the confidence (i.e., probability)
            idx = int(detections[0, 0, i, 1])   # extract the index of the class label
            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            if conf > confidence and idx == personIdx and not np.array_equal(prevDetections[cam], detections):
                # then compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                startX=max(1, startX)
                startY=max(1, startY)
                endX=min(endX, W-1)
                endY=min(endY,H-1)
                xlen=endX-startX
                ylen=endY-startY
                xcen=int((startX+endX)/2)
                ycen=int((startY+endY)/2)
                boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                # adhoc "fix" for out of focus blobs close to the camera
                # out of focus blobs sometimes falsely detect -- insects walking on camera, etc.
                # In my real world use I have some static false detections, mostly under IR or mixed lighting -- hanging plants etc.
                # I put camera specific adhoc filters here based on (xlen,ylen,xcenter,ycenter)
                # TODO: come up with better way to do it, probably return (xlen,ylen,xcenter,ycenter) and filter at saving or Notify step.
                if float(xlen*ylen)/(W*H) > blobThreshold:     # detection filling too much of the frame is bogus
                   continue
                # display and label the prediction
                label = "{:.1f}%  C:{},{}  W:{} H:{}  UL:{},{}  LR:{},{} {}".format(conf * 100,
                        str(xcen), str(ycen), str(xlen), str(ylen), str(startX), str(startY), str(endX), str(endY), dnnTarget)
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, label, (2, (H-5)-(28)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                personDetected = True
                initialConf=conf
                break   # one is enough
        prevDetections[cam]=detections
        if personDetected:  # Always zoom and verify
            personDetected = False  # repeat on zoomed detection box
            DNNdetect=True
            try:
                ## removing this box expansion really hurt the verification sensitivity
                ## removing this box expansion really hurt the verification sensitivity
                # zoom in on detection box and run second inference for verification.
                blen=max(xlen,ylen)
                if blen < PREPROCESS_DIMS[0]:
                    blen = PREPROCESS_DIMS[0]   # expand crop pixels so resize always makes smaller image
                adj=int(1.3*blen/2) # enlarge detection box by 30% and make crop be square about box center
                CstartX=max(xcen-adj,0)
                CendX=min(xcen+adj,W-1)
                CstartY=max(ycen-adj,0)
                CendY=min(ycen+adj,H-1)
                zimg = cv2.resize(orig_image[CstartY:CendY, CstartX:CendX], PREPROCESS_DIMS, interpolation = cv2.INTER_AREA)
            except Exception as e:
                print(" ocvdnn CPU AI verification crop region ERROR: ", startY, endY, startX, endX)
                continue
            (h, w) = zimg.shape[:2]
            blob = cv2.dnn.blobFromImage(zimg, 0.007843, PREPROCESS_DIMS, 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                conf = detections[0, 0, i, 2]
                idx = int(detections[0, 0, i, 1])
                if not np.array_equal(prevDetections, detections) and idx == 15:
                    if conf > verifyConf:
                        text = "Verify: {:.1f}%".format(conf * 100)   # show verification confidence
                        cv2.putText(image, text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        personDetected = True
                        break
            cfps.update()    # update the FPS counter
        # pass to yolo for verification, or pass as zoomed image for alerts
        try:
            # Queue results
            if yoloQ is not None:
                # pass to yolo  for verification, or pass as zoomed image for alerts
                if personDetected: # ocvdnn detection
                    detect+=1
                    if blen < __VERIFY_DIMS__[0]:
                        adj=int(1.1*__VERIFY_DIMS__[0]/2) 
                        CstartX=max(Xcenter-adj,0)
                        CendX=min(Xcenter+adj,W-1)
                        CstartY=max(Ycenter-adj,0)
                        CendY=min(Ycenter+adj,H-1)
                    person_crop = orig_image[CstartY:CendY, CstartX:CendX]
                    if yoloQ.full():
                        [_,_,_,_,_,_,_]=yoloQ.get(False)  # remove oldest result 
                        dcnt+=1                                               
                    yoloQ.put((image, cam, personDetected, imageDT, aiStr, boxPoints, person_crop.copy() ), True, 1.0)    # try not to drop frames with detections
                else:
                    noDetect+=1
                    if results.full():
                        [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                        ncnt+=1                       
                    if DNNdetect: # DNN verification failed
                        DNN_verify_fail+=1
                        results.put((image.copy(), cam, False, imageDT, aiStr, (-1,0, 0,0, 0,0, 0,0), None), zimg.copy(), 1.00) # -1 flags this AI verify fail
                    else:
                        results.put((image.copy(), cam, False, imageDT, aiStr, (0,0, 0,0, 0,0, 0,0), None), True, 0.200) # 0 boxpoints flag no detection
            else:
                if personDetected:
                    detect+=1
                    if results.full():
                        [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                        dcnt+=1                       
                    person_crop = image[CstartY:CendY, CstartX:CendX] # since no yolo verification, show original detection in zoomed version
                    results.put((image.copy(), cam, personDetected, imageDT, aiStr, boxPoints, person_crop.copy() ), True, 1.0) 
                else:
                    noDetect+=1
                    if results.full():
                        [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                        ncnt+=1                       
                    if DNNdetect: # DNN verification failed
                        DNN_verify_fail+=1
                        results.put((image.copy(), cam, False, imageDT, aiStr, (-1,0, 0,0, 0,0, 0,0), None), zimg.copy(), 1.00)
                    else:
                        results.put((image.copy(), cam, False, imageDT, aiStr, (0,0, 0,0, 0,0, 0,0), None), True, 0.200)
        except Exception as e:
            # presumably outptut queue was full, main thread too slow.
            ecnt+=1
            print("ocvdnn output queue results.put() Exception: " + str(e))
            continue

    # Thread exits
    cfps.stop()    # stop the FPS counter timer
    print("openCV SSDv1 dnn module CPU AI thread" + str(tnum) + ", waited: " + str(waits) + " dropped: " + str(dcnt+ncnt+ecnt) + " out of "
         + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))
    print("    openCV SSDv1 dnn module thread" + str(tnum) + " " + str(detect) + " Persons Detected.  " + str(noDetect) + " frames with no person.")
    print("    " + aiStr + " " + str(DNN_verify_fail) + " detections failed zoom-in verification.")
    print("    " + aiStr + " Detections dropped: " + str(dcnt) + " results dropped: " + str(ncnt) + " results.put() exceptions: " + str(ecnt))




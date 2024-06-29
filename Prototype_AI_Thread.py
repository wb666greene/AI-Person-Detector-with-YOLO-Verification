import numpy as np
import cv2
import datetime
from PIL import Image
from io import BytesIO
from imutils.video import FPS

# global to signal thread exit
global __Thread__
__Thread__ = False

global __VERIFY_DIMS__
__VERIFY_DIMS__ = (300,300)



'''
one time gode to run when thread is launched.
'''
def threadInit():
    return



'''
AI model dependent function to do the inference.
This is the function you need to write to add a new AI model option to the sysem,
along with changes in AI.py to add command line options for it and start and stop the thread function.
Search for #$$$# string to show where changes need to be made in analogy with the TPU thread, which is the
most straightforward to start and stop.

called as:
   image, personDetected, boxPoints, detectConfidence = do_inference( image, model, PREPROCESS_DIMS, confidence, blobThreshold )
'''
def do_inference( image, model, PREPROCESS_DIMS, confidence, blobThreshold ):


    boxPoints=(0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength
    personDetected = False
    # code to do an inference
    return image, personDetected, boxpoints, detectConfidence



'''
This should be pure "boilerplate" with no changes necessary
'''
def AI_thread(results, inframe, model, tnum, cameraLock, nextCamera, Ncameras,
                PREPROCESS_DIMS, confidence, verifyConf, dnnStr, blobThreshold, yoloQ):
    global __protoThread__
    global __VERIFY_DIMS__

    waits=0
    drops=0
    fcnt=0
    detect=0
    noDetect=0
    TPU_verify_fail=0
    if tnum > 0:
        aiStr = dnnStr + str(tnum)
    else:
        aiStr = dnnStr
    threadInit()
    print(aiStr + " AI thread is running...")
    if yoloQ is not None:
        print("    " + aiStr + " AI thread is using yolo verification.")

    __Thread__ = True
    cfps = FPS().start()
    while __Thread__ is True:
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
        personDetected = False
        TPUdetect=False
        # image is straignt from the camera, we draw boxes and labels on it later
        (H,W)=image.shape[:2]
        # orig_image is a copy of the image and is never drawn on, can be passed in the output queue if you don't want annotations.
        orig_image=image.copy()   # for zoomed in yolo verification

        # run the inference
        image, personDetected, boxPoints, detectConfidence = do_inference( image, model, PREPROCESS_DIMS, confidence, blobThreshold )

        fcnt+=1
        cfps.update()    # update the FPS counter
       # Next zoom in and repeat inference to verify detection
        ## removing this puts too much load on the much slower yolo thread,
        ## as this verification rejects a lot of plants as people detection.
        if personDetected:   # always verify now.
            try:    # repeat the inference zoomed in on the person detected
                TPUdetect=True
                personDetected = False
                ## removing this box expansion really hurt the verification sensitivity
                startX, startY, endX, endY, Xcenter, Ycenter, xlen, ylen = boxPoints
                blen=max(xlen,ylen)
                if blen < PREPROCESS_DIMS[0]:
                    blen = PREPROCESS_DIMS[0]   # expand crop pixels so resize always makes smaller image
                adj=int(1.3*blen/2) # enlarge detection box by 30% and make crop be square about box center
                CstartX=max(Xcenter-adj,0)
                CendX=min(Xcenter+adj,W-1)
                CstartY=max(Ycenter-adj,0)
                CendY=min(Ycenter+adj,H-1)
                zimg = cv2.resize(orig_image[CstartY:CendY, CstartX:CendX], PREPROCESS_DIMS, interpolation = cv2.INTER_AREA)
                (h, w) = zimg.shape[:2]  # this will be PREPROCESS_DIMS (300, 300)
                if (h,w) != PREPROCESS_DIMS:
                    print(" AI verification, Bad resize!  h:{}  w:{}".format(h, w))
                    continue
            except Exception as e:
                print("AI crop region Exception: " + str(e))
                ##print(" Coral crop region ERROR: {}:{} {}:{}  Cam:{}".format( str(startY), str(endY), str(startX), str(endX), str(cam) ) )
                continue
            # run inference on the zoomed in image, the minus one for blobThreshold signals don't want boxpoints or image annotations.
            zzimg, personDetected, _, detectConfidence = do_inference( zimg, model, PREPROCESS_DIMS, verifyConf, -1.0 )
            if personDetected:
                text = "Verify: {:.1f}%".format(detectConfidence * 100)   # show verification confidence on detection image
                cv2.putText(image, text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cfps.update()    # update the FPS counter
        # pass to yolo v4 for verification, or pass as zoomed image for alerts
        try:
            # Queue results
            if yoloQ is not None:
                # pass to yolo  for verification, or pass as zoomed image for alerts
                if personDetected: # detection
                    detect+=1        
                    if blen < __VERIFY_DIMS__[0]:
                        adj=int(1.1*__VERIFY_DIMS__[0]/2) 
                        CstartX=max(Xcenter-adj,0)
                        CendX=min(Xcenter+adj,W-1)
                        CstartY=max(Ycenter-adj,0)
                        CendY=min(Ycenter+adj,H-1)
                    person_crop = orig_image[CstartY:CendY, CstartX:CendX]
                    yoloQ.put((image.copy(), cam, personDetected, imageDT, aiStr, boxPoints, person_crop.copy() ), True, 1.0)    # try not to drop frames with detections
                else:
                    noDetect+=1
                    if TPUdetect: # TPU verification failed
                        TPU_verify_fail+=1
                        # So I could view the SSD initial detections that failed verification, also needs debug code in AI.py
                        results.put((image.copy(), cam, False, imageDT, aiStr, (-1,0, 0,0, 0,0, 0,0), zzimg.copy()), True, 1.00) # -1 flags TPU verify fail
                    else:  # No initial TPU detection
                        results.put((image.copy(), cam, False, imageDT, aiStr, (0,0, 0,0, 0,0, 0,0), None), True, 0.200) # 0 boxpoints flag no detection
            else:   # No yolo verification
                if personDetected:
                    detect+=1
                    person_crop = image[CstartY:CendY, CstartX:CendX] # since no yolo verification, show original detection in zoomed version
                    results.put((image.copy(), cam, personDetected, imageDT, aiStr, boxPoints, person_crop.copy() ), True, 1.0) # person_crop is zoom image here, instead of yolo frame
                else:
                    noDetect+=1
                    if TPUdetect: # TPU verification failed
                        TPU_verify_fail+=1
                        results.put((image.copy(), cam, False, imageDT, aiStr, (-1,0, 0,0, 0,0, 0,0), zzimg.copy() ), True, 1.00)  
                    else:
                        results.put((image.copy(), cam, False, imageDT, aiStr, (0,0, 0,0, 0,0, 0,0), None), True, 0.200)  #don't waste time wating for space to send null results
        except Exception as e:
            # presumably outptut queue was full, main thread too slow.
            ##print("Coral TPU queue write Exception: " + str(e))
            drops+=1
            continue
    # Thread exits
    cfps.stop()    # stop the FPS counter timer
    print(aiStr + " thread" + str(tnum) + ", waited: " + str(waits) + " dropped: " + str(drops) + " out of "
         + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))
    print("    " + aiStr + str(detect) + " Persons Detected.  " + str(noDetect) + " frames with no person.")
    print("   " + aiStr + " " + str(TPU_verify_fail) + " Detections failed zoom-in verification.")



'''
    16SEP2022wbk Reorganize code to follow Prototype_AI_Thread organization as verification of the prototype layout.
'''
# setup code to run at import Coral_TPU_Thread
import numpy as np
import cv2
import datetime
from PIL import Image
from io import BytesIO
from imutils.video import FPS


global aiStr
global __PYCORAL__
global __Thread__
__Thread__ = False

global __VERIFY_DIMS__
__VERIFY_DIMS__ = (300,300)

global __DEBUG__
__DEBUG__ = False


try:
    from edgetpu.detection.engine import DetectionEngine
    from edgetpu import __version__ as edgetpu_version
    __PYCORAL__ = 0
except ImportError:
    ### On 20.04 with PyCoral 2.0 runtime(14) either trying 15 first or 16 first works and never hits the
    ### [INFO]: PyCoral v.1x is not installed, trying v.1y" exception stanza.
    ### Situation is the same on 16.04 with PyCoral 1.0.1 runtime(13), Why did I think I needed to do this?
    print("[INFO]: Edgetpu support not installed, trying PyCoral")
    try:
        from pycoral.adapters.common import input_size
        from pycoral.adapters.detect import get_objects
        from pycoral.utils.dataset import read_label_file
        from pycoral.utils.edgetpu import make_interpreter
        from pycoral.utils.edgetpu import run_inference
        from pycoral.utils.edgetpu import get_runtime_version
        edgetpu_version=get_runtime_version()
        __PYCORAL__ = 16    #deb version 16.0
        print("PyCoral version: " + __import__("pycoral").__version__)
    except ImportError:
        print("[INFO]: PyCoral v.16 is not installed, trying v.15")
        try:
            from pycoral.adapters import common, detect
            from pycoral.utils.dataset import read_label_file
            from pycoral.utils.edgetpu import make_interpreter, get_runtime_version
            edgetpu_version=get_runtime_version()
            __PYCORAL__ = 15    # deb version 15.0
            print("PyCoral version: " + __import__("pycoral").__version__)
        except ImportError:
            print("[ERROR]: PyCoral TPU support is not installed, exiting ...")
            quit()

print('Edgetpu_api version: ' + edgetpu_version)



'''
one time gode to run when thread is launched.
'''
def threadInit():
    return



"""
# AI model dependent function to do the inference
# image, personDetected, boxpoints, detectConfidence = do_inference( image, PREPROCESS_DIMS, confidence )
def do_inference( image, model, PREPROCESS_DIMS, confidence, blobThreshold ):

    boxPoints=(0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength
    personDetected = False

    return image, personDetected, boxpoints, detectConfidence
"""
def do_inference( image, model, PREPROCESS_DIMS, confidence, blobThreshold ):

        global aiStr
        boxPoints=(0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength
        personDetected = False
        (H,W)=image.shape[:2]
        frame_rgb = cv2.cvtColor(cv2.resize(image, PREPROCESS_DIMS), cv2.COLOR_BGR2RGB)
        if __PYCORAL__ == 0:
            frame = Image.fromarray(frame_rgb)
            if edgetpu_version < '2.11.2':
                detection = model.DetectWithImage(frame, threshold=confidence, keep_aspect_ratio=True, relative_coord=False)
            else:
                detection = model.detect_with_image(frame, threshold=confidence, keep_aspect_ratio=True, relative_coord=False)
        else:   # using pycoral
            if __PYCORAL__ == 15:
                frame = Image.fromarray(frame_rgb)
                common.set_input(model,frame)
                model.invoke()
                detection=detect.get_objects(model, confidence, (1.0,1.0))
            else:
                run_inference(model, frame_rgb.tobytes())
                detection=get_objects(model, confidence, (1.0,1.0))
        # loop over the detection results
        detectConfidence = 0.0
        for r in detection:
            found=False
            if __PYCORAL__ == 0:
                if r.label_id == 0: # coco "person" lable index, should this be passsed in parameter?
                    # extract the bounding box for predicted class label
                    box = r.bounding_box.flatten().astype("int")
                    (startX, startY, endX, endY) = box.flatten().astype("int")
                    found=True
            else:
                if r.id == 0:   # coco "person" lable index
                    startX=r.bbox.xmin
                    startY=r.bbox.ymin
                    endX=r.bbox.xmax
                    endY=r.bbox.ymax
                    found=True
            if found:
                detectConfidence = r.score
                if blobThreshold >= 0:   # <0 is signal that we don't want boxpoints or image annotations
                    X_MULTIPLIER = float(W) / PREPROCESS_DIMS[0]
                    Y_MULTIPLIER = float(H) / PREPROCESS_DIMS[1]
                    startX = int(startX * X_MULTIPLIER)
                    startY = int(startY * Y_MULTIPLIER)
                    endX = int(endX * X_MULTIPLIER)
                    endY = int(endY * Y_MULTIPLIER)
                    xlen=endX-startX
                    ylen=endY-startY
                    if float(xlen*ylen)/(W*H) > blobThreshold:     # detection filling too much of the frame is bogus
                        continue
                    xcen=int((startX+endX)/2)
                    ycen=int((startY+endY)/2)
                    boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                    # draw the bounding box and label on the image
                    cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 1)
                    label = "{:.1f}%  C:{},{}  W:{} H:{}  UL:{},{}  LR:{},{} {}".format(detectConfidence * 100,
                        str(xcen), str(ycen), str(xlen), str(ylen), str(startX), str(startY), str(endX), str(endY), aiStr)
                    cv2.putText(image, label, (2, (H-5)-28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                personDetected = True
                break   # one person detection is enoughinit
        return image, personDetected, boxPoints, detectConfidence




'''
This should be pure "boilerplate" with no changes necessary
'''
def AI_thread(results, inframe, model, tnum, cameraLock, nextCamera, Ncameras,
                PREPROCESS_DIMS, confidence, verifyConf, dnnStr, blobThreshold, yoloQ):
    global __Thread__
    global aiStr
    global __VERIFY_DIMS__
    global __DEBUG__
    aiStr=dnnStr
    waits=0
    fcnt=0
    detect=0
    noDetect=0
    TPU_verify_fail=0
    dcnt=0
    ncnt=0
    ecnt=0
    threadInit()
    print( aiStr + " AI thread" + " is running...")
    if yoloQ is not None:
        print("    TPU AI thread is using yolo verification.")

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
        ## as this verification rejects a lot of plants as people detection
        ## still need to repeat it to reduce load the slower yolo verification
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
                    print(" Coral TPU verification, Bad resize!  h:{}  w:{}".format(h, w))
                    continue
            except Exception as e:
                print("Coral TPU crop region Exception: " + str(e))
                ##print(" Coral crop region ERROR: {}:{} {}:{}  Cam:{}".format( str(startY), str(endY), str(startX), str(endX), str(cam) ) )
                continue

            # run inference on the zoomed in image, the minus one for blobThreshold signals don't want boxpoints or image annotations.
            zzimg, personDetected, _, detectConfidence = do_inference( zimg, model, PREPROCESS_DIMS, verifyConf, -1.0 )

            if personDetected:
                text = "Verify: {:.1f}%".format(detectConfidence * 100)   # show verification confidence on detection image
                cv2.putText(image, text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cfps.update()    # update the FPS counter
        try:
            # Queue results
            if yoloQ is not None:
                # pass to yolo  for verification, or pass as zoomed image for alerts
                if personDetected: # TPU detection
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
                    yoloQ.put((image.copy(), cam, personDetected, imageDT, aiStr, boxPoints, person_crop.copy() ), True, 1.0)    # try not to drop frames with detections
                else:
                    noDetect+=1
                    if results.full():
                        [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                        ncnt+=1                       
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
                    if results.full():
                        [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                        dcnt+=1                       
                    results.put((image.copy(), cam, personDetected, imageDT, aiStr, boxPoints, person_crop.copy() ), True, 1.0) # person_crop is zoom image here, instead of yolo frame
                else:
                    noDetect+=1
                    if results.full():
                        [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                        ncnt+=1                       
                    if TPUdetect: # TPU verification failed
                        TPU_verify_fail+=1
                        results.put((image.copy(), cam, False, imageDT, aiStr, (-1,0, 0,0, 0,0, 0,0), zzimg.copy() ), True, 1.00)  
                    else:
                        results.put((image.copy(), cam, False, imageDT, aiStr, (0,0, 0,0, 0,0, 0,0), None), True, 0.200)  #don't waste time wating for space to send null results
        except Exception as e:
            # presumably outptut queue was full, main thread too slow.
            ##print("Coral TPU results.put() Exception: " + str(e))
            ecnt+=1
            continue
    # Thread exits
    cfps.stop()    # stop the FPS counter timer
    print(aiStr + ", waited: " + str(waits) + " dropped: " + str(dcnt+ncnt+ecnt) + " out of "
         + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))
    print("   " + aiStr + " " + str(detect) + " Persons Detected.  " + str(noDetect) + " frames with no person.")
    print("   " + aiStr + " " + str(TPU_verify_fail) + " TPU detections failed zoom-in verification.")
    print("   " + aiStr + " Detections dropped: " + str(dcnt) + " results dropped: " + str(ncnt) + " results.put() exceptions: " + str(ecnt))


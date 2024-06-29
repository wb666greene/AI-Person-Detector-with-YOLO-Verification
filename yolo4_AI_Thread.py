#import argparse
#import os
#import glob
import random
import darknet
import time
import cv2
import numpy as np
from imutils.video import FPS

'''
    conda activate pycoral
    python AI.py -y4AI -d 1 -cam 6onvif.txt -rtsp 19cams.rtsp
    
    On my i9-12900K with GTX 3070 GPU running yolov4-608.cfg model
    I get ~34 fps per second on 25 cameras.
    Performace is great, so far no false positive detections and seems
    to have greater detection sensitivity, espcially at night.
    
    I do get segfault crashes from a darknet function
    that I've not had any luck tracking down so far.  It may be a GPU
    memory issue as it seems to correlate with other code using the GPU/display.
    
    At this point it is so frustrating that I may shitcan the Darknet yolo4.
    Especially since the Ultralytics yolo8 works so well.
'''

global __Thread__
__Thread__ = False

global network
global class_names
global class_colors


'''
one time code to run when thread is launched.
'''
def threadInit():
    '''
    eventually, should pass in PREPROCESS_DIMS and choose
    either the 608, 512, or 416 yolo4 model, just use 608 for now
    but smaller should be better for weaker machines, but bigger seems better for accuracy.
    '''
    global network
    global class_names
    global class_colors
    random.seed(33)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        "./yolo4/yolov4-608.cfg",
        "./yolo4/coco.data",
        "./yolo4/yolov4.weights",
        batch_size=1
    )



'''
AI model dependent function to do the inference.
This is the function you need to write to add a new AI model option to the sysem,
along with changes in AI.py to add command line options for it and start and stop the thread function.
Search for #$$$# string to show where changes need to be made in analogy with the TPU thread, which is the
most straightforward to start and stop.

called as:
   image, personDetected, boxpoints, detectConfidence = do_inference( image, model, PREPROCESS_DIMS, confidence, blobThreshold )
'''
def do_inference( input_image, model, PREPROCESS_DIMS, confidence, blobThreshold ):

    global network
    global class_names
    global class_colors
    boxPoints=(0,0, 0,0, 0,0, 0,0)  # (startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength)
    personDetected = False
    detectConfidence = 0.0
    # code to do an inference
    #%%%#print('DN', end='', flush=True)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    #%%%#print('*DD', end='', flush=True) # debug trace to see where segfault happens
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=confidence-0.001)
    #%%%#print('*F', end='', flush=True)
    darknet.free_image(darknet_image)
    #%%%#print('*DB', end='', flush=True)
    '''
        Note that I modified darknet.draw_boxes() to return the number of persons detected,
        and their cv2 style boxPoints in input_image coordinates.  I let darknet draw all the objects detected
        mostly for grins at the moment, but I only return the person detection boxPoints.
        Only the first (highest confidence) person detection is used, but having multiple persons
        might end up being useful eventually.  My SSD code stops with the first above threshold detection.
        Could do it here too, but I'm exploring yolo as much as trying to actually use it.
    '''
    image, personDetected, boxpoints, detectConfidence = darknet.draw_boxes(detections, image_rgb, class_colors, network)
    #%%%#print('!', end='', flush=True)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if personDetected:
        ## dump some data to help me figure out what is what.
        ##darknet.print_detections(detections, coordinates=True)
        ##print(boxpoints)
        (H,W)=image.shape[:2]
        ### this doesn't seem right:
        ###detectConfidence = float(detections[-1][1])/100     # sorted highest confidence last!
        ### modified darknet.draw_boxes() to return person detection confidence as float instead of string
        if blobThreshold >= 0:   # <0 is signal that we don't want boxpoints or image annotations on zoomed verification image
            startX, startY, endX, endY = boxpoints[-1]
            xlen=endX-startX
            ylen=endY-startY
            xcen=int((startX+endX)/2)
            ycen=int((startY+endY)/2)
            boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
            ### dump some data to help me figure out what is what.
            ###darknet.print_detections(detections, coordinates=True)
            ###print(boxPoints)
            if float(xlen*ylen)/(W*H) > blobThreshold:     # detection filling too much of the frame is bogus
                personDetected = False
    #%%%#print('#', end='', flush=True)
    return image.copy(), personDetected, boxPoints, detectConfidence



'''
This should be pure "boilerplate" with no or minimal changes necessary
'''
def AI_thread(results, inframe, model, tnum, cameraLock, nextCamera, Ncameras,
                PREPROCESS_DIMS, confidence, verifyConf, dnnStr, blobThreshold, yoloQ):
    global __Thread__

    waits=0
    dcnt=0
    ncnt=0
    ecnt=0
    fcnt=0
    detect=0
    noDetect=0
    yolo_verify_fail=0
    if tnum > 0:
        aiStr = dnnStr + str(tnum)
    else:
        aiStr = dnnStr
    threadInit()
    print(aiStr + " AI thread is running...")

    if yoloQ is not None:
        print("    " + aiStr + " yolo4 AI thread doesn't use yolo queue! Ignoring.")
        yoloQ = None

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
        # image is straignt from the camera, we draw boxes and labels on it later
        (H,W)=image.shape[:2]
        # orig_image is a copy of the image and is never drawn on, can be passed in the output queue if you don't want annotations.
        orig_image=image.copy()   # for zoomed in yolo verification

        # run the inference
        #%%%#print(str(cam), end='', flush=True) # debug trace to see where segfault happens
        yoloDetect=False
        img, personDetected, boxPoints, detectConfidence = do_inference( image.copy(), model, PREPROCESS_DIMS, confidence, blobThreshold )
        image=img.copy()
        #%%%#print('r', end='', flush=True) ## Crashing in this function
        fcnt+=1
        cfps.update()    # update the FPS counter
        # Next zoom in and repeat inference to verify detection
        ## removing this puts too much load on the much slower yolo thread,
        ## as this verification rejects a lot of plants as people detection.
        if personDetected:   # always verify now.
            try:    # repeat the inference zoomed in on the person detected
                personDetected = False
                yoloDetect=True
                ## removing this box expansion really hurt the verification sensitivity
                startX, startY, endX, endY, xcen, ycen, xlen, ylen = boxPoints
                label = "{:.1f}%  C:{},{}  W:{} H:{}  UL:{},{}  LR:{},{} {}".format(detectConfidence * 100,
                        str(xcen), str(ycen), str(xlen), str(ylen), str(startX), str(startY), str(endX), str(endY), aiStr)
                cv2.putText(image, label, (2, (H-5)-28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                # zoom in on detection box and run second inference for verification.
                blen=max(xlen,ylen)
                if blen < PREPROCESS_DIMS[0]:
                    blen = PREPROCESS_DIMS[0]   # expand crop pixels so resize always makes smaller image
                adj=int(1.2*blen/2) # enlarge detection box and make crop be square about box center
                CstartX=max(xcen-adj,0)
                CendX=min(xcen+adj,W-1)
                CstartY=max(ycen-adj,0)
                CendY=min(ycen+adj,H-1)
                zimg = orig_image[CstartY:CendY, CstartX:CendX]
            except Exception as e:
                print("Yolo4 crop region Exception: " + str(e))
                ##print(" Yolo4 crop region ERROR: {}:{} {}:{}  Cam:{}".format( str(startY), str(endY), str(startX), str(endX), str(cam) ) )
                continue

            # run inference on the zoomed in image, the minus one for blobThreshold signals don't want boxpoints or image annotations.
            #%%%#print('V', end='', flush=True) # debug trace to see where segfault happens
            # pass to yolo v4 for verification, or pass as zoomed image for alerts
            zzimg, personDetected, _, detectConfidence = do_inference( zimg.copy(), model, PREPROCESS_DIMS, verifyConf, -1.0 )
            cfps.update()    # update the FPS counter
            """ #%%%#
            if personDetected:
                print('d', end='', flush=True) # debug trace to see where segfault happens
            else:
                print('n', end='', flush=True) # debug trace to see where segfault happens
            """
            if personDetected:
                text = "Verify: {:.1f}%".format(detectConfidence * 100)   # show verification confidence on detection image
                cv2.putText(image, text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # Queue results
        try:
            if personDetected:
                detect+=1
                if results.full():
                    [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                    dcnt+=1                       
                results.put((image.copy(), cam, personDetected, imageDT, aiStr, boxPoints, zzimg.copy() ), True, 1.0) # yolo_frame is person_crop here
            else:
                noDetect+=1
                if results.full():
                    [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                    ncnt+=1                       
                if yoloDetect:
                    yolo_verify_fail+=1
                    results.put((image.copy(), cam, False, imageDT, aiStr, (-1,0, 0,0, 0,0, 0,0), zzimg.copy() ), True, 1.00)  # -1 boxpoints flags zoom verification failed
                else:
                    results.put((image.copy(), cam, False, imageDT, aiStr, (0,0, 0,0, 0,0, 0,0), None ), True, 0.200)  # 0 boxpoints flag no initial detection
            """ #%%%#
            if (noDetect + detect) % Ncameras:
                print('|', end='', flush=True) # debug trace to see where segfault happens
            else:
            print('|', end='\n', flush=True) # debug trace to see where segfault happens
            """
        except:
            # presumably outptut queue was full, main thread too slow.
            if personDetected:
                print("Person detection dropped!  Cam" + str(cam) + imageDT.strftime("%Y-%m-%d_%H:%M:%S.%f"), flush=True)
            ##else:     # verified that quickly dropping no detection frames helps prevent dropping detecting frames, but otherwise not helpful
            ##    print("Dropped a no detection frame." + imageDT.strftime("%Y-%m-%d_%H:%M:%S.%f"), flush=True)
            ecnt+=1
            continue
    # Thread exits
    cfps.stop()    # stop the FPS counter timer
    print(aiStr + " thread" + str(tnum) + ", waited: " + str(waits) + " dropped: " + str(dcnt+ncnt+ecnt) + " out of "
         + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))
    print("    " + aiStr + " " + str(detect) + " Persons Detected.  " + str(noDetect) + " frames with no person.")
    print("    " + aiStr + " " + str(yolo_verify_fail) + " detections failed zoom-in verification.")
    print("    " + aiStr + " Detections dropped: " + str(dcnt) + " results dropped: " + str(ncnt) + " results.put() exceptions: " + str(ecnt))



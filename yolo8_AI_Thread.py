from ultralytics import YOLO
import datetime
from imutils.video import FPS
import cv2


'''
    conda activate yolo8
    python AI.py -y8AI -d 1 -cam 6onvif.txt -rtsp 18cams.rtsp
    
    On my i9-12900K with GTX 3070 GPU running yolov8x.pt model
    I get ~33 fps per second on 24 cameras.
    Performace is great, so far and seems to have greater detection sensitivity, espcially at night.
    
    One false positive detection of the neighbor's cat by the pool.
'''


# global to signal thread exit
global __Thread__
__Thread__ = False

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



'''
AI model dependent function to do the inference.
This is the function you need to write to add a new AI model option to the sysem,
along with changes in AI.py to add command line options for it and start and stop the thread function.
Search for #$$$# string to show where changes need to be made in analogy with the TPU thread, which is the
most straightforward to start and stop.

called as:
   image, personDetected, boxPoints, detectConfidence = do_inference( image, model, PREPROCESS_DIMS, confidence, blobThreshold )
'''
def do_inference( image, model, confidence, blobThreshold ):
    
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
                if blobThreshold >= 0.0:    # negative blobThreshold signals verification only, don't want boxpoints or blob rejection
                    startX = int(boxes.data[i][0].item())
                    startY = int(boxes.data[i][1].item())
                    endX = int(boxes.data[i][2].item())
                    endY = int(boxes.data[i][3].item())
                    xlen=endX-startX
                    ylen=endY-startY
                    xcen=int((startX+endX)/2)
                    ycen=int((startY+endY)/2)
                    boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                    (H,W)=image.shape[:2]
                    if float(xlen*ylen)/(W*H) > blobThreshold:     # detection filling too much of the frame is bogus
                        personDetected = False
                break
    #print(type(annotated_image), annotated_image.shape)
    return annotated_image.copy(), personDetected, boxPoints, detectConfidence



'''
This should be pure "boilerplate" with no changes necessary
'''
def AI_thread(results, inframe, Xmodel, tnum, cameraLock, nextCamera, Ncameras,
                PREPROCESS_DIMS, confidence, verifyConf, dnnStr, blobThreshold, yoloQ):
    global __Thread__
    global model
    
    waits=0
    fcnt=0
    ecnt=0
    dcnt=0
    ncnt=0
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
        print("    " + aiStr + " yolo8 AI thread doesn't use yolo queue! Ignoring.")
        yoloQ = None  # we need this to be none to skip irrevalent code later.

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
        yoloDetect=False
        img, personDetected, boxPoints, detectConfidence = do_inference( image, model, confidence, blobThreshold )
        #print(type(img), img.shape)
        fcnt+=1
        cfps.update()    # update the FPS counter
       # Next zoom in and repeat inference to verify detection
        ## removing this puts too much load on the much slower yolo thread,
        ## as this verification rejects a lot of plants as people detection.
        if personDetected:   # always verify now.
            try:    # repeat the inference zoomed in on the person detected
                yoloDetect=True
                personDetected = False
                ## removing this box expansion really hurt the verification sensitivity
                startX, startY, endX, endY, Xcenter, Ycenter, xlen, ylen = boxPoints
                label = "{:.1f}%  C:{},{}  W:{} H:{}  UL:{},{}  LR:{},{} {}".format(detectConfidence * 100,
                        str(Xcenter), str(Ycenter), str(xlen), str(ylen), str(startX), str(startY), str(endX), str(endY), aiStr)
                cv2.putText(img, label, (2, (H-5)-28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                # zoom in on detection box and run second inference for verification.
                # change verification "zoom" selection to make square crop symmetrical about Xcenter,Ycenter
                blen=max(xlen,ylen)
                if blen < PREPROCESS_DIMS[0]:
                    blen = PREPROCESS_DIMS[0]   # expand crop pixels so resize always makes smaller image
                adj=int(1.2*blen/2) # enlarge detection box and make crop be square about box center
                CstartX=max(Xcenter-adj,0)
                CendX=min(Xcenter+adj,W-1)
                CstartY=max(Ycenter-adj,0)
                CendY=min(Ycenter+adj,H-1)
                zimg = orig_image[CstartY:CendY, CstartX:CendX]
            except Exception as e:
                print("Yolo8 crop region Exception: " + str(e))
                ##print(" Coral crop region ERROR: {}:{} {}:{}  Cam:{}".format( str(startY), str(endY), str(startX), str(endX), str(cam) ) )
                continue

            # run inference on the zoomed in image, the minus one for blobThreshold signals don't want boxpoints or image annotations.
            zzimg, personDetected, _, detectConfidence = do_inference( zimg, model, verifyConf, -1.0 )
            cfps.update()    # update the FPS counter
            if personDetected:
                text = "Verify: {:.1f}%".format(detectConfidence * 100)   # show verification confidence on detection image
                cv2.putText(img, text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # Queue results
        try:
            if personDetected:
                detect+=1
                if results.full():
                    [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                    dcnt+=1                       
                results.put((img.copy(), cam, personDetected, imageDT, aiStr, boxPoints, zzimg.copy() ), block=True, timeout=1.0) # yolo_frame is person_crop here
            else:
                noDetect+=1
                if results.full():
                    [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                    ncnt+=1                       
                if yoloDetect:
                    yolo_verify_fail+=1
                    results.put((img.copy(), cam, False, imageDT, aiStr, (-1,0, 0,0, 0,0, 0,0), zzimg.copy() ), True, 1.00)  # -1 boxpoints flags zoom verification failed
                else:
                    results.put((img.copy(), cam, False, imageDT, aiStr, (0,0, 0,0, 0,0, 0,0), None ), True, 0.200)  # 0 boxpoints flag no initial detection


                results.put((img.copy(), cam, False, imageDT, aiStr, boxPoints, None), block=True, timeout=0.200)  #don't waste time wating for space to send null results
        except Exception as e:
            # presumably outptut queue was full, main thread too slow.
            if personDetected:
                print("Person detection dropped!  Cam" + str(cam) + imageDT.strftime("%Y-%m-%d_%H:%M:%S.%f"), flush=True)
            ecnt+=1
            print("Yolo8 queue.put() Exception: " + str(e)) # e doesn't print anything
            continue
    # Thread exits
    cfps.stop()    # stop the FPS counter timer
    print(aiStr + " thread waited: " + str(waits) + " dropped: " + str(dcnt+ncnt+ecnt) + " out of "
         + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))
    print("   " + aiStr + " " + str(detect) + " Persons Detected.  " + str(noDetect) + " frames with no person.")
    print("   " + aiStr + " " + str(yolo_verify_fail) + " detections failed zoom-in verification.")
    print("   " + aiStr + " Detections dropped: " + str(dcnt) + " results dropped: " + str(ncnt) + " results.put() exceptions: " + str(ecnt))


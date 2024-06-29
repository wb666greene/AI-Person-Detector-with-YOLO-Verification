"""
    Darknet yolo4 verification thread.
    Basically an intermediate queue is created and filled with zoomed detections
    from the TPU, CPU, etc. AI and then darknet yolo4 is run on the zoomed in
    image and if person is detected the image is sent to the results queue.

    On 17-8750H laptop with Nvidia GTX1060:
    python3 AI.py -nTPU 1 -y4v -d 1 -cam 6onvif.txt -rtsp 19cams.rtsp
    Yielded ~68 fps with 25 cameras for an ~69683 second test run.
    There were ~4.7 million frames processed by the TPU with 11280 persons detected.
    The yolo4 verification accepted 10953 and rejected 327.
    My review suggests almost all the rejections were false negatives, but a fair price to pay
    for the near complete rejection of false positives from my bogus detection collection.
    One real false positive that was rejected:
    an image where a dog was detected as a person, and person walking the dog had not yet entered frame.

"""

##import argparse
##import os
##import glob
import random
import darknet
import time
import cv2
import numpy as np
# why imported twice? bug or valid reason?
# some Google suggests its a bug, but nothing bad happens.
#import darknet
global __darknetThread__
__darknetThread__ = False
global __verifyConf__
__verifyConf__ = 0.65


def image_detection(input_image, network, class_names, class_colors, thresh):
#def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    #image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    #darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    '''
        Note that I modified darknet.draw_boxes() to return the number of persons detected,
        and their cv2 style boxPoints in input_image coordinates.  I let darknet draw all the objects detected
        mostly for grins at the moment, but I only return the person detection boxpoints.
        Only the first (highest confidence) person detection is used, but having multiple persons
        might end up being useful eventually.  My SSD code stops with the first above threshold detection.
        Could do it here too, but I'm exploring yolo as much as trying to actually use it.
    '''
    image, persons_detected, boxpoints, person_conf = darknet.draw_boxes(detections, image_rgb, class_colors, network)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections, persons_detected, boxpoints, person_conf





def yolov4_thread(results, yoloQ):
    global __darknetThread__
    global __verifyConf__

    print("Starting Yolo v4 verification thread...")
    if yoloQ is None:
        print(    "ERROR! no yolo Queue!")

    random.seed(33)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        "./yolo4/yolov4-608.cfg",
        "./yolo4/coco.data",
        "./yolo4/yolov4.weights",
        batch_size=1
    )
   
    yoloVerified=0
    yoloRejected=0
    yoloWaited = 0
    dcnt=0
    ecnt=0
    ncnt=0
    print("Yolo v4 verification thread is running...")
    __darknetThread__ = True

    while __darknetThread__ is True:
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
            image, detections, persons_detected, _, person_conf = image_detection(yolo_frame, network, class_names, class_colors, __verifyConf__)
            if persons_detected >= 1:   # yolov4 has verified the MobilenetSSDv2 person detection
                ## image is the yolo_frame with the yolo detection boxes overlaid (from the boxpoints).
                ## boxPoints are from the SSD inference.
                # draw the verification confidence on the  ssd_frame
                yoloVerified+=1
                text = "Yolo4: {:.1f}%".format(person_conf * 100)   # show verification confidence on detection image
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
            print('[Exception] yolov4_thread'+ str(cam) + ': ' + str(e))
            continue
    print("Yolo v4 frames Verified: {}, Rejected: {},  Waited: {} seconds.".format(str(yoloVerified), str(yoloRejected), str(yoloWaited)))
    print("    Verified dropped: " + str(dcnt) + " results dropped: " + str(ncnt) + " results.put() exceptions: " + str(ecnt))


#! /usr/bin/python3

# Typical command lines for a full test on 22.04 with yolo8 verification:
"""
wally@Wahine:~$ conda activate yolo8
(yolo8) wally@Wahine:~$ cd AI
(yolo8) wally@Wahine:~/AI$    python AI.py -y8v -nTPU 1 -d 1 -cam 6onvif.txt -rtsp 19cams.rtsp
"""

## appending 2> /dev/null to the start command line helps if you are getting "Invalid UE golomb code" opencv warnings.
## it also gets rid of mouch of the darknet library "chatter" on startup.

'''
     4SEP2022wbk -- unify AI_dev.py and AI_yolo.py to produce this AI.py code.
                    Start project to add yolo7 verification option, and yolo7 AI inference thread

    13SEP2022wbk    --yolo4_verify option added,  Fisheye camera support from TPU.py merged.
                    Restored support for NCS2, openCV dnn thread, and SSDv1.  But these are really
                    only useful for systems with maybe four cameras or less, and a higher false positive tolerence
                    unless NVIDIA GPU is available to use -y4v or -y8v.  Something like an old "gaming laptop"
                    with i5 CPU and GTX970 class GPU.  OpenVINO CPU often gives better frame rate than NCS2 on i7.

    15SEP2022wbk    Tested, but bug remains: typing "q  key" in openCV window segfaults when -y4v (--yolo4_verify) option is active.
                    So remove openCV keypress exit option "it hurts when I do this, then don't do it!" But since the design
                    is expected to not have a live display, sending SIGINT to the process (Ctrl-C in the terminal window)
                    is the expected way to exit the program.

    16SEP2022wbk    Make TPU thread follow the Prototype_AI_Thread format.
                    Unify the thread parameters across all the AI models.

    9SEP2023wbk     Drop the idea of adding yolo7 and use the newer easier to use and better documented yolo8
                    https://github.com/ultralytics/ultralytics

    30APR2023wbk    Seems impossible to get Yolo8 to run on 16.04 becasue of glibc version issues, but darknet Yolo4 runs fine.

    2MAY2023wbk     Make debug code in in main() and AI threads that lets me see what was being rejected in verification steps be command line parameter.
                    Lower verifyConf by 0.05 for -y4AI and -y8AI, seem to have too many false negatives with the TPU/OpenVINO verification default value.
'''
#
#
# Historical evolution
### AI_dev.py 13JUL2019wbk
#
### 16APR2021wbk
#       Modified TPU support to try the "legacy" edgetpu API and if its not found try the new PyCoral API
#       Tested on Ubuntu 20.04 i3-4025 CPU with PyCoral and the new MPCIe TPU module (< half the cost of USB3 TPU)
#       Verified on Ubuntu 16.04 i7 desktop using "legacy" edgeTPU API
#
## 10DEC2019wbk
# Increase queue depth to 2, test if queue full, read and discard oldest to make room for newest
#
## 11DEC2019wbk, add PiCamera Module support, change some command argument defaults and names.
## 27DEC2019wbk, tested PiCamera Module support on Pi3B with NCS and OpenVINO:
#  ./AI_dev.py -nNCS 1 -pi -ls --> get ~3.8 fps
# And Coral TPU:
#  ./AI_dev.py -nTPU 1 -pi -ls --> get ~8.0 fps
#
## 9MAY2020wbk add intermediate yolov4 thread to verify detections.
#  ./AI_yolo.py -nTPU 1 -d 1 -rtsp cam/4HD_Amcrest16.rtsp
#
## TODO:
''' 10JUL2022wbk
 --> DONE   1) remove NCS/NSC2 and NCS sdkv1 support, NCS2 support restored 13SEP2022.
 --> DONE   2) add yolo verification to CPU/GPU threads.
 --> DONE   3) remove PiCam support as no PiCamera module runs on a system with Nvidia
                GPU, unless PiCameara hardware and software is compatible with Jetson Nano.
                A bit of Google suggests the hardware is compatible, but need to use the openCV
                gstreamer capture routines instead of the picamera Python module, so my code is useless as is.

    14SEP2022wbk
            1) Should the blobThreshold be an array with one value for each camera?  Not needed so far.

    2MAY2023wbk
 --> DONE   1) Investigate adding a camera name along with the cameraURL in -cam and -rtsp files.
            2) Investigate adding a yolo or other verification AI model using TPU or NCS2 for CUDA incapable systems.

    7MAY2023wbk
            1) Consider removing NCS2/OpenVINO support, tests with -nt 1 -nNCS 1 -nTPU1  show the vast majority
                of detections are wiht the TPU and it is not just that the TPU processes more frames, they just detect
                fewer people, even with their thresholds reduced by 0.1
'''
#
#
''' Some performance test results.

    On 17-8750H laptop with Nvidia GTX1060:
    python3 AI.py -nTPU 1 -y4v -d 1 -cam 6onvif.txt -rtsp 19cams.rtsp
    Yielded ~68 fps with 25 cameras for an ~69683 second test run.
    There were ~4.7 million frames processed by the TPU with 11280 persons detected.
    The yolo4 verification accepted 10953 and rejected 327.
    My review suggests almost all the rejections were false negatives, but a fair price to pay
    for the near complete rejection of false positives from my bogus detection collection.
    One real false positive that was rejected:
    an image where a dog was detected as a person, and person walking the dog had not yet entered frame.


    On my i9-12900K with GTX 3070 GPU using yolov4-608.cfg darknet model:
    conda activate pycoral (or yolo8)
    python AI.py -y4AI -d 1 -cam 6onvif.txt -rtsp 19cams.rtsp
    I get ~32 fps per second on 25 cameras.
    2697289 images processed, 13960 Persons Detected.  2683329 frames with no person.
    22592 detections failed zoom-in verification.
    Performace is great, so far no false positive detections and seems
    to have great detection sensitivity, espcially at night.
    I do get segfault crashes from a darknet function
    that I've not had any luck tracking down so far.  It may be a GPU
    memory issue as it seems to correlate with other code using the GPU/display.
    At this point it is so frustrating that I may shitcan the Darknet yolo4.
    Especially since the Ultralytics yolo8 works so well.

    On my i9-12900K with GTX 3070 GPU using yolov8x.pt model
    conda activate yolo8
    python AI.py -y8AI -d 1 -cam 6onvif.txt -rtsp 18cams.rtsp
    I get ~33 fps per second on 24 cameras.
    Performace is great, so far and seems to have great detection sensitivity, espcially at night.
    One false positive detection of the neighbor's cat by the pool.


    It is clear that the "zoom in on detection and re-inference" rejects a large number of false positives.
    On a recent test run with i7-8750H and Nvidia GTX1060:
    conda activate yolo8
    python3 AI.py -d 0 -z -nTPU 1 -cam 6onvif.txt -y8v -rtsp 19cams.rtsp
    3878202 images were processed from 25 cameras  netting 72.39 inferences/sec (a bit under 3 fps per camera)
    11041 Persons Detected with TPU verification, while 29494 TPU detections failed zoom-in verification.
    A very large percentage of the "zoom-in and re-inference" detections were plants, pets, and other true negatives.
    Yolo8 Verified: 10907, Rejected: 134. A review of the rejects suggests they were all false negatives since
    a person was in the image, but this ~1.2% false negative rate is an acceptable price to pay for the near
    elimination false positives on my collection of false positives images collected over a couple of years
    without the yolo verification step.
'''



# import the necessary packages
import sys
import signal
from imutils.video import FPS
import argparse
import numpy as np
import cv2
import paho.mqtt.client as mqtt
import os
import time
import datetime
import requests
from PIL import Image
from io import BytesIO

# threading stuff
from queue import Queue
from threading import Lock, Thread
# for saving PTZ view maps
import pickle


# *** System Globals
# these are write once in main() and read-only everywhere else, thus don't need syncronization
global QUIT
QUIT=False  # True exits main loop and all threads
global Nrtsp
global Nonvif
global Ncameras
global __CamName__
global AlarmMode    # would be Notify, Audio, or Idle, Idle mode doesn't save detections
global UImode
global CameraToView
global subscribeTopic
subscribeTopic = "Alarm/#"  # topic controller publishes to to set AI operational modes
global Nmqtt
global mqttCamOffset
global inframe
global mqttFrameDrops
global mqttFrames
global mqttCamsOneThread
# this variable to distribute queued data to the AI threads needs syncronization
global nextCamera
nextCamera = 0      # next camera queue for AI threads to use to grab a frame
cameraLock = Lock()
global SSDv1
# globals for thread control

global __onvifThread__
global __rtspThread__
global __fisheyeThread__

global __yolo4Verify__
global __yolo8Verify__
global __yolo4Thread__
global __yolo8Thread__

global GRID_SIZE
global CLIP_LIMIT
global CLAHE

global __DEBUG__
__DEBUG__ = False

# *** constants for MobileNet-SSD & MobileNet-SSD_V2  AI models
# frame dimensions should be sqaure for MobileNet-SSD
PREPROCESS_DIMS = (300, 300)


if 1:
    # *** get command line parameters
    # construct the argument parser and parse the arguments for this module
    ap = argparse.ArgumentParser()

    # enable zoom and verify using yolo inference (requires Nvidia cuda capable video card.
    ap.add_argument("-y4v", "--yolo4_verify", action="store_true", help="Verify detection with darknet yolov4 inference on zoomed region")
    ap.add_argument("-y8v", "--yolo8_verify", action="store_true", help="Verify detection with a yolov8 inference on zoomed region")
    # OpenVINO GPU yolo4 verification
    ap.add_argument("-y4ovv", "--yolo4ov_verify", action="store_true", help="Verify detection with OpenVINO yolo4 inference on zoomed region")
    ap.add_argument("-myriad", "--MYRIAD", action="store_true", help="Verify detection with OpenVINO yolo4 using NCS2 instead of GPU")
    ap.add_argument("-yvq", "--YoloVQ", type=int, default=10, help="Depth of YOLO verification queue, should be about YOLO framerate, default=10")
    ap.add_argument("-rq", "--resultsQ", type=int, default=10, help="Minimum Depth of results queue, default=10")

    # enable a yolo thread for inference.
    ap.add_argument("-y4AI", "--yolov4", action="store_true", help="enable darknet yolov4 inference thread")
    ap.add_argument("-y8AI", "--yolov8", action="store_true", help="enable ultralytics yolov8 inference thread")

    # parameters that might be installation dependent
    ap.add_argument("-c", "--confidence", type=float, default=0.70, help="detection confidence threshold")
    ap.add_argument("-vc", "--verifyConfidence", type=float, default=0.80, help="detection confidence for verification")
    ap.add_argument("-yvc", "--yoloVerifyConfidence", type=float, default=0.75, help="detection confidence for verification")
    ap.add_argument("-blob", "--blobFilter", type=float, default=0.33, help="reject detections that are more than this fraction of the frame")

    # specify number of Coral TPU sticks
    ap.add_argument("-nTPU", "--nTPU", type=int, default=0, help="number of Coral TPU devices")

    # use one mqtt thread for all cameras instead of one mqtt thread per mqtt camera
    ap.add_argument("-mqttMode", "--mqttCamOneThread", action="store_true", help="Use one mqtt thread for all mqtt cameras")
    ap.add_argument("-mqttDemand", "--mqttDemand", action="store_true", help="Use sendOne/N handshake for MQTT cameras")

    # number of software (CPU only) AI threads, always have one thread per installed NCS stick
    ap.add_argument("-nt", "--nAIcpuThreads", type=int, default=0, help="0 --> no CPU AI thread, >0 --> N threads")
    ap.add_argument("-GPU", "--useGPU", action="store_true", help="use GPU instead of CPU AI thread, forces N threads == 1")

    # must specify number of NCS sticks for OpenVINO, trying load in a try block and error, wrecks the system!
    ap.add_argument("-nNCS", "--nNCS", type=int, default=0, help="number of Myraid devices")

    # use Mobilenet-SSD Caffe model instead of Tensorflow Mobilenet-SSDv2_coco
    ap.add_argument("-SSDv1", "--SSDv1", action="store_true", help="Use original Mobilenet-SSD Caffe model for NCS & OVcpu")

    # specify text file with list of URLs for camera rtsp streams
    ap.add_argument("-rtsp", "--rtspURLs", default="cameraURL.rtsp", help="path to file containing rtsp camera stream URLs")

    # specify text file with list of URLs cameras http "Onvif" snapshot jpg images
    ap.add_argument("-cam", "--cameraURLs", default="cameraURL.txt", help="path to file containing http camera jpeg image URLs")

    # display mode, mostly for test/debug and setup, general plan would be to run "headless"
    ap.add_argument("-d", "--display", type=int, default=1,
        help="display images on host screen, 0=no display, 1=live display")

    # specify MQTT broker
    ap.add_argument("-mqtt", "--mqttBroker", default="localhost", help="name or IP of MQTT Broker")

    # specify MQTT broker for camera images via MQTT, if not "localhost"
    ap.add_argument("-camMQTT", "--mqttCameraBroker", default="localhost", help="name or IP of MQTTcam/# message broker")
    # number of MQTT cameras published as Topic: MQTTcam/N, subscribed here as Topic: MQTTcam/#, Cams numbered 0 to N-1
    ap.add_argument("-Nmqtt", "--NmqttCams", type=int, default=0,
                    help="number of MQTT cameras published as Topic: MQTTcam/N,  Cams numbered 0 to N-1")
    # alternate, specify a list of camera numbers
    ap.add_argument("-camList", "--mqttCamList", type=int, nargs='+',
                    help="list of MQTTcam/N subscription topic numbers,  cam/N numbered from 0 to Nmqtt-1.")

    # specify display width and height
    ap.add_argument("-dw", "--displayWidth", type=int, default=3840, help="host display Width in pixels, default=1920")
    ap.add_argument("-dh", "--displayHeight", type=int, default=2160, help="host display Height in pixels, default=1080")

    # specify host display width and height of camera image
    ap.add_argument("-iw", "--imwinWidth", type=int, default=640, help="camera host display window Width in pixels, default=640")
    ap.add_argument("-ih", "--imwinHeight", type=int, default=360, help="camera host display window Height in pixels, default=360")

    # specify file path of location to same detection images on the localhost
    ap.add_argument("-sp", "--savePath", default="", help="path to location for saving detection images, default ../detect")
    # save all processed images, fills disk quickly, really slows things down, but useful for test/debug

    ## CLAHE parameters
    ap.add_argument("-cl", "--ClipLimit", type=float, default=4.5, help="CLAHE clipLimit parameter, default=4.5")
    ap.add_argument("-gs", "--GridSize", type=int, default=5, help="CLAHE tileGridSize parameter, default=5")
    ap.add_argument("-clahe", "--CLAHE", action="store_true", help="Enable CLAHE contrast enhancement on zoomed detection")

    # debug visulize verification rejections
    ap.add_argument("-dbg", "--debug", action="store_true", help="enable debug display of verification failures")

    # show zoom image of detections even if -d parameter is 0
    ap.add_argument("-z", "--zoom", action="store_true", help="always display zoomed image of detection.")
    
    # Disable local save of detections on AI host -nls is same as -nsz and -nsf options
    ap.add_argument("-nls", "--NoLocalSave", action="store_true", help="no saving of detection images on local AI host")
    # don't save zoomed image locally
    ap.add_argument("-nsz", "--NoSaveZoom", action="store_true", help="don't locally save zoomed detection image")
    # don't save full images locally  
    ap.add_argument("-nsf", "--NoSaveFull", action="store_true", help="don't locally save full detection frame.")

    # send full frame image of detections to node-red instead of zoomed in on detection
    ap.add_argument("-nrf", "--nodeRedFull", action="store_true", help="full frame detection images to node-read instead of zoom images")

    args = vars(ap.parse_args())


    mqttCamsOneThread = args["mqttCamOneThread"]
    MQTTdemand = args["mqttDemand"]


# mark start of this code in log file
print("$$$**************************************************************$$$")
currentDT = datetime.datetime.now()
print("*** " + currentDT.strftime(" %Y-%m-%d %H:%M:%S") + "  ***")
print("[INFO] using openCV-" + cv2.__version__)


# *** Function definitions
#**********************************************************************************************************************
#**********************************************************************************************************************
#**********************************************************************************************************************

# Boilerplate code to setup signal handler for graceful shutdown on Linux
def sigint_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        #print('caught SIGINT, normal exit. -- ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        QUIT=True

def sighup_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGHUP! ** ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        QUIT=True

def sigquit_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGQUIT! *** ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        QUIT=True

def sigterm_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGTERM! **** ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        QUIT=True

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGHUP, sighup_handler)
signal.signal(signal.SIGQUIT, sigquit_handler)
signal.signal(signal.SIGTERM, sigterm_handler)



#**********************************************************************************************************************
## MQTT callback functions
##
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    global subscribeTopic
    #print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.  -- straight from Paho-Mqtt docs!
    client.subscribe(subscribeTopic)



# The callback for when a PUBLISH message is received from the server, aka message from SUBSCRIBE topic.
def on_message(client, userdata, msg):
    global AlarmMode    # would be Notify, Audio, or Idle, Idle mode doesn't save detections
    global UImode
    global CameraToView
    if str(msg.topic) == "Alarm/MODE":          # Idle will not save detections, Audio & Notify are the same here
        currentDT = datetime.datetime.now()     # logfile entry
        AlarmMode = str(msg.payload.decode('utf-8'))
        print(str(msg.topic)+":  " + AlarmMode + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S"))
        return
    # UImode: 0->no Dasboard display, 1->live image from selected cameram 2->detections from selected camera, 3->detection from any camera
    if str(msg.topic) == "Alarm/UImode":    # dashboard control Disable, Detections, Live exposes apparent node-red websocket bugs
        currentDT = datetime.datetime.now() # especially if browser is not on localhost, use sparingly, useful for camera setup.
        print(str(msg.topic)+": " + str(int(msg.payload)) + currentDT.strftime("   ... %Y-%m-%d %H:%M:%S"))
        UImode = int(msg.payload)
        return
    if str(msg.topic) == "Alarm/ViewCamera":    # dashboard control to select image to view
        currentDT = datetime.datetime.now()
        print(str(msg.topic)+": " + str(int(msg.payload)) + currentDT.strftime("   ... %Y-%m-%d %H:%M:%S"))
        CameraToView = int(msg.payload)
        return


def on_publish(client, userdata, mid):
    #print("mid: " + str(mid))      # don't think I need to care about this for now, print for initial tests
    pass


def on_disconnect(client, userdata, rc):
    if rc != 0:
        currentDT = datetime.datetime.now()
        print("Unexpected MQTT disconnection!" + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S  "), client)
    pass


# callbacks for mqttCam that can't be shared
# mqttCamsOneThread=False is default
## True/False no significant difference on i7-6700K Desktop both ~53 fps for 15 ~5 fps MQTTcams from i5ai rtsp2mqtt server
               ## On Pi4, XU-4 etc.  one thread for all mqttCams is ~1.5 fps faster.
if not mqttCamsOneThread:   # use one mqtt thread per mqttCam
# callbacks for mqttCam that can't be shared
  def on_mqttCam_connect(client, userdata, flags, rc):
        camT=userdata[0]
        camN=userdata[1]
        client.subscribe("MQTTcam/"+str(camT), 0)


  def on_mqttCam(client, userdata, msg):
    global mqttCamOffset
    global inframe
    global mqttFrameDrops
    global mqttFrames
    # put input image into the camera's inframe queue
    try:
        camT=userdata[0]
        camN=userdata[1]
        mqttFrames[camN]+=1
        # thanks to @krambriw on the node-red user forum for clarifying this for me
        npimg=np.frombuffer(msg.payload, np.uint8)      # convert msg.payload to numpy array
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)   # decode image file into openCV image
        imageDT=datetime.datetime.now()
        if inframe[camN+mqttCamOffset].full():
            [_,_,_]=inframe[camN+mqttCamOffset].get(False)
            mqttFrameDrops[camN]+=1     # is happes here, shouldn't happen below
        inframe[camN+mqttCamOffset].put((frame, camN+mqttCamOffset, imageDT), False)
        ##inframe[camN+mqttCamOffset].put((frame, camN+mqttCamOffset), True, 0.200)
    except:
        mqttFrameDrops[camN]+=1     # queue.full() is not 100% reliable
    if MQTTdemand:
        client.publish(str("sendOne/" + str(camT)), "", 0, False)
##    time.sleep(0.001)     # force thread dispatch, hard to tell if this helps or not.
    return

else:
  def on_mqttCam_connect(client, camList, flags, rc):
     for camN in camList:
        client.subscribe("MQTTcam/"+str(camN), 0)


  def on_mqttCam(client, camList, msg):
    global mqttCamOffset
    global inframe
    global mqttFrameDrops
    global mqttFrames
    global Nmqtt    ## eliminate len(camList) call by using global
    if msg.topic.startswith("MQTTcam/"):
        camNstr=msg.topic[len("MQTTcam/"):]    # get camera number as string
        if camNstr.isdecimal():
            camT = int(camNstr)
            if camT not in camList:
                currentDT = datetime.datetime.now()
                print("[Error! Invalid MQTTcam Camera number: " + str(camT) + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S"))
                return
            for i in range(Nmqtt):
                if camT == camList[i]:
                    camN=i
                    break
        else:
            currentDT = datetime.datetime.now()
            print("[Error! Invalid MQTTcam message sub-topic: " + camNstr + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S"))
            return
        # put input image into the camera's inframe queue
        try:
            mqttFrames[camN]+=1
            # thanks to @krambriw on the node-red user forum for clarifying this for me
            npimg=np.frombuffer(msg.payload, np.uint8)      # convert msg.payload to numpy array
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)   # decode image file into openCV image
            imageDT=datetime.datetime.now()
            if inframe[camN+mqttCamOffset].full():
                [_,_,_]=inframe[camN+mqttCamOffset].get(False)
                mqttFrameDrops[camN]+=1     # is happes here, shouldn't happen below
            inframe[camN+mqttCamOffset].put((frame, camN+mqttCamOffset, imageDT), False)
        except:
            mqttFrameDrops[camN]+=1     # queue.full() is not 100% reliable
        try:
            if MQTTdemand:
                client.publish(str("sendOne/" + str(camT)), "", 0, False)
##            time.sleep(0.001)     # force thread dispatch, hard to tell if this helps or not.
        except Exception as e:
            print("pub error " + str(e))
        return


'''
# Hard to believe but Python threads don't have a terminate signal, need a kludge like this
# There are other ways, but I want some stats printed at thread termination.
# I think the real issue is the AI threads are in seperate Python files and its a scope issue for the Quit global
def QUITf():
    global QUIT
    return QUIT
'''


# *** main()
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def main():
    global QUIT
    global AlarmMode    # would be Notify, Audio, or Idle, Idle mode doesn't save detections
    AlarmMode="Audio"   # will be Email, Audio, or Idle  via MQTT controller from alarmboneServer
    global CameraToView
    CameraToView=0
    global UImode
    UImode=0    # controls if MQTT buffers of processed images from selected camera are sent as topic: ImageBuffer
    global subscribeTopic
    global Nonvif
    global Nrtsp
    global Nmqtt
    global mqttCamOffset
    global mqttFrameDrops
    global inframe
    global Ncameras
    global __CamName__
    global CamName
    global mqttFrames
    global mqttCamsOneThread
    global __PYCORAL__
    # globals for thread control, maybe QUITf() was cleaner, but I can stage the stopping for better [INFO} reporting on exit

    global __darknetThread__
    global __rtspThread__
    global __fisheyeThread__
    # command line "store true" flags
    global __yolo4Verify__
    ##global __yolo7Verify__
    ##global __yolo4Thread__
    global SSDv1

    global GRID_SIZE
    global CLIP_LIMIT
    global CLAHE

    global __DEBUG__


    # set variables from command line auguments or defaults
    #$$$# active TPU
    nCoral = args["nTPU"]
    if nCoral > 1:
        nCoral = 1      # Not finished multiple TPU support, not sure it is needed or will be useful.
    nCPUthreads = args["nAIcpuThreads"]
    # It appears that Intel GPU and CUDA cannot be used together.
    useGPU = args["useGPU"]
    if useGPU and nCPUthreads !=1:
        nCPUthreads=1
    nNCSthreads = args["nNCS"]   # the same thread function is used for OpenVINO CPU and NCS/NCS2 threads, my naming could be better.
    SSDv1 = args["SSDv1"]
    if SSDv1 and nNCSthreads >0:
        print("[INFO] NCS2 does not support SSDv1 Caffe model, switching to CPU model.")
        nNCSthreads = 0
        nCPUthreads += 1
    confidence = args["confidence"]
    verifyConf = args["verifyConfidence"]
    yoloVerifyConf = args["yoloVerifyConfidence"]
    blobThreshold = args["blobFilter"]
    MQTTcameraServer = args["mqttCameraBroker"]
    Nmqtt = args["NmqttCams"]
    camList=args["mqttCamList"]
    if camList is not None:
        Nmqtt=len(camList)
    elif Nmqtt>0:
        camList=[]
    for i in range(Nmqtt):
        camList.append(i)
    dispMode = args["display"]
    if dispMode > 1:
        displayMode=1
    CAMERAS = args["cameraURLs"]
    RTSP = args["rtspURLs"]
    MQTTserver = args["mqttBroker"]     # this is for command and control messages, and detection messages
    displayWidth = args["displayWidth"]
    displayHeight = args["displayHeight"]
    imwinWidth = args["imwinWidth"]
    imwinHeight = args["imwinHeight"]
    savePath = args["savePath"]
    NoLocalSave = args["NoLocalSave"]
    NoSaveZoom = args["NoSaveZoom"]
    NoSaveFull = args["NoSaveFull"]
    nodeRedFull= args["nodeRedFull"]
    __DEBUG__ = args["debug"]
    show_zoom = args["zoom"]

    # *** setup path to save AI detection images
    if savePath == "":
        home, _ = os.path.split(os.getcwd())
        detectPath = home + "/detect"
        if os.path.exists(detectPath) == False:
            os.mkdir(detectPath)
    else:
        detectPath=savePath
        if os.path.exists(detectPath) == False:
            print(" Path to location to save detection images must exist!  Exiting ...")
            quit()
    yolo4_verify=args["yolo4_verify"]
    OVyolo4_verify=args["yolo4ov_verify"]
    OVmyriad=args["MYRIAD"]
    yoloVQdepth=args["YoloVQ"]
    if OVmyriad is True:
        yoloVQdepth=3   # NCS2 has ~2 fps frame rate with YOLO  
    resultsQdepth=args["resultsQ"]
    yolo8_verify=args["yolo8_verify"]
    yolo4AI=args["yolov4"]
    yolo8AI=args["yolov8"]


    # 24PR2023wbk: Screen for incompatable options and force sane behavior.
    # such as only one of -y4v or -y8v can be used.
    # only one of -y4AI or -y8AI can be used.
    # -y4v or -y8v  can not be used if either -y4AI or -y8AI is active.
    # Need to test if TPU, CPU, NCS2, can be mixed with one of -y4AI or -y8AI.
    if yolo4AI and yolo8AI:
        yolo4AI = False
        print("[WARN] Only one of -y4AI or -y8AI can be used.  Forcing -y8AI")
    if yolo4_verify and yolo8_verify and OVyolo4_verify:
        yolo4_verify=False
        OVyolo4_verify = False
        print("[WARN] Only one of -y4v or -y4ovv or -y8v can be used. Forcing -y8v")
    if yolo4AI or yolo8AI:
        yolo4_verify = False
        yolo8_verify = False
        OVyolo4_verify = False
    if useGPU and (yolo4_verify or yolo8_verify or yolo4AI or yolo8AI):
        useGPU=False
        nCPUthreads=1   # does multiple GPU threads make sense? I make no attempt allow both GPU and CPU threads
        print("[WARN] Intel openCL GPU can not be used with CUDA, using CPU AI thread instead.")


    # init CLAHE
    CLAHE = args["CLAHE"]
    if CLAHE:
        GRID_SIZE = (args["GridSize"],args["GridSize"])
        CLIP_LIMIT = args["ClipLimit"]
        clahe = cv2.createCLAHE(CLIP_LIMIT,GRID_SIZE)


    # *** get Onvif camera URLs
    # cameraURL.txt file can be created by first running the nodejs program (requires node-onvif be installed):
    # nodejs onvif_discover.js
    #
    # This code does not really use any Onvif features, Onvif compatability is useful to "automate" getting  URLs used to grab snapshots.
    # Any camera that returns a jpeg image from a web request to a static URL should work.
    CamName=list()  # dynamically built list of camera names read from file or created as Cam0, Cam1, ... CamN
    try:
        #CameraURL=[line.rstrip() for line in open(CAMERAS)]    # force file not found
        #Nonvif=len(CameraURL)
        l=[line.split() for line in open(CAMERAS)]
        CameraURL=list()
        for i in range(len(l)):
            CameraURL.append(l[i][0])
            if len(l[i]) > 1:
                CamName.append(l[i][1])
            else:
                CamName.append("Cam" + str(i))
        Nonvif=len(CameraURL)
        print("[INFO] " + str(Nonvif) + " http Onvif snapshot threads will be created.")
    except Exception as e:
        # No Onvif cameras
        #print(e)
        print("[INFO] No " + str(CAMERAS) + " file.  No Onvif snapshot threads will be created.")
        Nonvif=0
    Ncameras=Nonvif
    #print(CamName)


    # *** get rtsp URLs
    try:
        #rtspURL=[line.rstrip() for line in open(RTSP)]
        #Nrtsp=len(rtspURL)
        rtspURL=list()
        l=[line.split() for line in open(RTSP)]
        for i in range(len(l)):
            rtspURL.append(l[i][0])
            if len(l[i]) > 1:
                CamName.append(l[i][1])
            else:
                CamName.append("Cam" + str(i+Ncameras))
        Nrtsp=len(rtspURL)
        print("[INFO] " + str(Nrtsp) + " rtsp stream threads will be created.")
    except:
        # no rtsp cameras
        print("[INFO] No " + str(RTSP) + " file.  No rtsp stream threads will be created.")
        Nrtsp=0
    Ncameras+=Nrtsp


    # define fisheye cameras and virtual PTZ views
    # fisheye.rtsp is expected to be created with the interactive fisheye_window C++ utility program
    try:
        l=[line.rstrip() for line in open('fisheye.rtsp')]
        FErtspURL=list()
        PTZparam=list()
        j=-1
        for i in range(len(l)):
            if not l[i]: continue
            if l[i].startswith('rtsp'):
                FErtspURL.append(l[i])
                j+=1
                PTZparam.append([])
            else:
                PTZparam[j].append(l[i].strip().split(' '))

        print("[INFO] Setting up PTZ virtual cameras views from fisheye camera ...")
        #print(FErtspURL)
        #print(PTZparam)
        Nfisheye=len(FErtspURL)     # modified rtsp thread will send PTZ views to seperate queues, this is number of fisheye threads
        NfeCam=0                    # total number of queues to be created for virtual PTZ cameras
        for i in range(Nfisheye):
            if len(PTZparam[i])<2 or len(PTZparam[i][0])<2 or len(PTZparam[i][1])!=6:
                # this is where Python's features make code simple but obtuse!
                # setting up this data structure in C/C++ gives me cooties with the variable number of possible PTZ views per camera!
                print('[ERROR] PTZparam[' + str(i) + '] must contain [srcW, srcH],[dstW,detH,  alpha,beta,theta,zoom] entries, Exiting ...')
                quit()
            NfeCam += len(PTZparam[i])-1 # the first entry is camera resolution, not a PTZ view
        # I'm not bothering with naming fisheye camera views, just create sequential names
        for i in range(NfeCam):
            CamName.append("FEview" + str(i))
    except:
        # no fisheye cameras
        print("[INFO] No fisheye.rtsp file.  No fisheye camera rtsp stream threads will be created.")
        NfeCam=0
        Nfisheye=0
    FishEyeOffset=Ncameras
    Ncameras+=NfeCam # add fisheye virtual PTZ views to cameras count


    mqttCamOffset = Ncameras
    mqttFrameDrops = 0
    mqttFrames = 0
    if Nmqtt > 0:
        print("[INFO] allocating " + str(Nmqtt) + " MQTT image queues...")
        # Again not trying to name MQTT cams, may remove this eventually or use IMGMQ transport instead
        for i in range(Nmqtt):
            CamName.append("MQTT" + str(i))
    Ncameras+=Nmqtt     # I generally expect Nmqtt to be zero if Ncameras is not zero at this point, but its not necessary
    if Ncameras == 0:
        print("[INFO] No Cameras, rtsp Streams, or MQTT image inputs specified!  Exiting...")
        quit()


    # *** allocate queues
    print("[INFO] allocating camera and stream image queues...")
    # we simply make one queue for each camera, rtsp stream, and MQTTcamera
    QDEPTH = 3      # Make queue depth be three, sometimes get two frames less then 20 mS appart with
                    # "read queue if full and then write frame to queue" in camera input thread
##    QDEPTH = 2      # bump up for trial of "read queue if full and then write to queue" in camera input thread
##    QDEPTH = 1      # small values improve latency
    results = Queue(max(resultsQdepth,Ncameras))
    inframe = list()
    for i in range(Ncameras):
        inframe.append(Queue(QDEPTH))

    if yolo4_verify or yolo8_verify or OVyolo4_verify:
        ###yoloQ = Queue(max(10,Ncameras))  # this can lead to very long latencies if the AI thread is much faster than the yolo verification thread.
        yoloQ = Queue(yoloVQdepth)   # This should be approx the lessor of the AI thread frame rate and yolo verification frame rate
    else:
        yoloQ = None

    # build grey image for mqtt windows
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (127,127,127)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])


    # *** setup display windows if necessary
    # mostly for initial setup and testing, not worth a lot of effort at the moment
    if dispMode > 0:
        if Nonvif > 0:
            print("[INFO] setting up Onvif camera image windows ...")
            for i in range(Nonvif):
                name=str("Live_" + CamName[i])
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if Nrtsp > 0:
            print("[INFO] setting up rtsp camera image windows ...")
            for i in range(Nrtsp):
                name=str("Live_" + CamName[i+Nonvif])
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if NfeCam > 0:
            print("[INFO] setting up  FishEye camera PTZ windows ...")
            for i in range(NfeCam):
                name=str("Live_" + CamName[i+FishEyeOffset])
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if Nmqtt > 0:
            print("[INFO] setting up MQTT camera image windows ...")
            for i in range(Nmqtt):
                name=str("Live_" + CamName[i+mqttCamOffset])
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.imshow(name, img)
                cv2.waitKey(1)
        # setup yolov4 verification windows
        if yolo4_verify or yolo8_verify:
            print("[INFO] setting up YOLO verification/reject image windows ...")
            cv2.namedWindow("yolo_verify", flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
            cv2.imshow("yolo_verify", img)
            cv2.waitKey(1)
            cv2.namedWindow("yolo_reject",flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
            cv2.imshow("yolo_reject", img)
            cv2.waitKey(1)
        else:
            print("[INFO] setting detection zoom image window ...")
            cv2.namedWindow("detection_zoom", flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
            cv2.imshow("detection_zoom", img)
            cv2.waitKey(1)

        # *** move windows into tiled grid
        top=20
        left=2
        Xborder=3   ## attempt to compensate for openCV window "decorations" varies too much with system to really work
        Yborder=32
        Xshift=imwinWidth+Xborder
        Yshift=imwinHeight+Yborder
        Nrows=int(displayHeight/Yshift)
        for i in range(Ncameras):
            name=str("Live_" + str(i))
            col=int(i/Nrows)
            row=i%Nrows
            cv2.moveWindow(name, left+col*Xshift, top+row*Yshift)
    else:
        if show_zoom:
            print("[INFO] setting detection zoom image window ...")
            cv2.namedWindow("detection_zoom", flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
            cv2.imshow("detection_zoom", img)
            cv2.waitKey(1)



    # *** connect to MQTT broker for control/status messages
    print("[INFO] connecting to MQTT " + MQTTserver + " broker...")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_publish = on_publish
    client.on_disconnect = on_disconnect
    client.will_set("AI/Status", "Python AI has died!", 2, True)  # let everyone know we have died, perhaps node-red can restart it
    client.connect(MQTTserver, 1883, 60)
    client.loop_start()


    # starting AI threads can take a long time, send image and message to dasboard to indicate progress
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (192,127,127)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    client.publish("ImageBuffer/!Starting AI threads, this can take awhile!.", bytearray(img_as_jpg), 0, False)


    if nCoral + nCPUthreads + nNCSthreads == 0 and not (yolo4AI or yolo8AI):
        print("[INFO] No Coral TPU, OpenVINO CPU or GPU devices specified,  forcing one CPU AI thread.")
        nCPUthreads=1   # we always can force one CPU thread, but ~1.8 seconds/frame on Pi3B+

    # these need to be loaded before an AI thread launches them
    if yolo4_verify:
        #import yolo4 darknet
        import darknet_Thread
        darknet_Thread.__verifyConf__ = yoloVerifyConf
    if yolo8_verify:
        #import Ultralytics yolo8 darknet
        import yolo8_verification_Thread
        # using yolov8x.pt for now m is  "fastest" x is "most accurate" l is in between
        yolo8_verification_Thread.__y8modelSTR__ = 'yolo8/yolov8x.pt'
        yolo8_verification_Thread.__verifyConf__ = yoloVerifyConf
    if OVyolo4_verify:
        import yolo4OpenvinoVerification_Thread
        yolo4OpenvinoVerification_Thread.__verifyConf__ = yoloVerifyConf
        if OVmyriad:
            yolo4OpenvinoVerification_Thread.__device__ = "MYRIAD"
            
    # *** setup and start Coral AI threads
    # Might consider moving this into the thread function.
    ### Setup Coral AI
    # initialize the labels dictionary
    if nCoral > 0:
        import Coral_TPU_Thread
        #$$$# import Prototype_AI_Thread  # TPU version of prototype thread for testing/debug
        print("[INFO] parsing mobilenet_ssd_v2 coco class labels for Coral TPU...")
        modelPath = "mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
        if Coral_TPU_Thread.__PYCORAL__ == 0:
        #$$$# if Prototype_AI_Thread.__PYCORAL__ is False:
            labels = {}
            for row in open("mobilenet_ssd_v2/coco_labels.txt"):
                # unpack the row and update the labels dictionary
                (classID, label) = row.strip().split(maxsplit=1)
                labels[int(classID)] = label.strip()
            print("[INFO] loading Coral mobilenet_ssd_v2_coco model...")
            model = Coral_TPU_Thread.DetectionEngine(modelPath)
            #$$$#model = Prototype_AI_Thread.DetectionEngine(modelPath)
        else:
            labels = Coral_TPU_Thread.read_label_file("mobilenet_ssd_v2/coco_labels.txt")
            model = Coral_TPU_Thread.make_interpreter(modelPath)    # if both installed can't predict which will be used.
            ##model = Coral_TPU_Thread.make_interpreter(modelPath, "usb")   # choose usb TPU if both installed
            ##model = Coral_TPU_Thread.make_interpreter(modelPath, "pci")   # use pci TPU if both installed
            #$$$# labels = Prototype_AI_Thread.read_label_file("mobilenet_ssd_v2/coco_labels.txt")
            #$$$# model = Prototype_AI_Thread.make_interpreter(modelPath)    # if both installed can't predict which will be used.
            model.allocate_tensors()

        # *** start Coral TPU threads
        Ct = list() ## not necessary only supporting a single TPU for now.
        print("[INFO] starting " + str(nCoral) + " Coral TPU AI Threads ...")
        for i in range(nCoral):
            print("... loading model...")
            #$$$# Ct.append(Thread(target=Prototype_AI_Thread.AI_thread,
            if __DEBUG__:
                Coral_TPU_Thread.__DEBUG__ = True
            if yolo8_verify:
                Coral_TPU_Thread.__VERIFY_DIMS__ = (640,640)
            if yolo4_verify or OVyolo4_verify:
                Coral_TPU_Thread.__VERIFY_DIMS__ = (608,608)
            Ct.append(Thread(target=Coral_TPU_Thread.AI_thread,
                args=(results, inframe, model, i, cameraLock, nextCamera, Ncameras,
                    PREPROCESS_DIMS, confidence, verifyConf, "TPU", blobThreshold,  yoloQ)))
            Ct[i].start()


    # *** setup and start Myriad OpenVINO
    ## Hmmm... single NCS, Caffe SSDv1 ~9.7 fps with 5 Onvif cameras,  TensorFlow SSDv2  gets only ~5.7 fps, with NCS2 ~11.8 fps
    ## NCS support removed from OpenVINO 2021.1
    if nNCSthreads > 0:
      if cv2.__version__.find("openvino") > 0:
        import OpenVINO_Thread
        if SSDv1:
            print("[INFO] loading Caffe Mobilenet-SSD model for OpenVINO Myriad NCS2 AI threads...")
            OVstr = "CaffeSSDncs"
        else:
            ## fragile works for 2021.1, need better way to detect openVINO version lacks NCS support and needs IR10 models
            if cv2.__version__ == "4.5.0-openvino" or cv2.__version__ == "4.5.1-openvino" or cv2.__version__ == "4.5.2-openvino":
                print("[INFO] loading Tensor Flow Mobilenet-SSD v2 FP16 IR10 model for OpenVINO_2021.1 Myriad NCS2 AI threads...")
                OVstr = "SSDv2_IR10ncs"
            else:
                print("[INFO] loading Tensor Flow Mobilenet-SSD v2 FP16 model for OpenVINO Myriad NCS2 AI threads...")
                OVstr = "SSDv2ncs"
        netOV=list()
        for i in range(nNCSthreads):
            print("... loading model...")
            if SSDv1:
                netOV.append(cv2.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD_deploy.prototxt", "MobileNetSSD/MobileNetSSD_deploy.caffemodel"))
                OpenVINO_Thread.SSDv1 = True
            else:
                ## fragile works for 2021.1, need better way to detect openVINO version lacks NCS support and needs IR10 models
                if cv2.__version__ == "4.5.0-openvino" or cv2.__version__ == "4.5.1-openvino" or cv2.__version__ == "4.5.2-openvino":
                    netOV.append(cv2.dnn.readNet("mobilenet_ssd_v2/MobilenetSSDv2cocoIR10.xml", "mobilenet_ssd_v2/MobilenetSSDv2cocoIR10.bin"))
                else:
                    netOV.append(cv2.dnn.readNet("mobilenet_ssd_v2/MobilenetSSDv2coco.xml", "mobilenet_ssd_v2/MobilenetSSDv2coco.bin"))
            netOV[i].setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
            netOV[i].setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)  # specify the target device as the Myriad processor on the NCS
        # *** start OpenVINO AI threads
        OVt = list()
        print("[INFO] starting " + str(nNCSthreads) + " OpenVINO Myriad NCS2 AI Threads ...")
        if yolo8_verify:
            OpenVINO_Thread.__VERIFY_DIMS__ = (640,640)
        if yolo4_verify or OVyolo4_verify:
            OpenVINO_Thread.__VERIFY_DIMS__ = (608,608)
        for i in range(nNCSthreads):
            OVt.append(Thread(target=OpenVINO_Thread.AI_thread,
                args=(results, inframe, netOV[i], i, cameraLock, nextCamera, Ncameras,
                    PREPROCESS_DIMS, confidence-0.1, verifyConf-0.1, OVstr, blobThreshold,  yoloQ)))    # these seem less sensitive
            OVt[i].start()
      else:
        print("[ERROR!] OpenVINO version of openCV is not active, check $PYTHONPATH")
        print(" No MYRIAD (NCS/NCS2) OpenVINO threads will be created!")
        nNCSthreads = 0
        if nCoral+nCPUthreads == 0:
            print("[INFO] No Coral TPU device or CPU threads specified, forcing one CPU AI thread.")
            nCPUthreads=1   # we always can force one CPU thread, but ~1.8 seconds/frame on Pi3B+


    # ** setup and start CPU/GPU AI threads, usually only one makes sense.
    ## TODO: do I want SSDv2 option for CPU threads as well??  Done, Made FP32 version with Model Optimizer.
    ## Will need to make FP32 version,  SSDv2 error: "Inference Engine backend: The plugin does not support FP16 in function 'initPlugin'"
    if nCPUthreads > 0:
        net=list()
        if cv2.__version__.find("openvino") > 0:
            import OpenVINO_Thread
            if SSDv1:
                print("[INFO] loading Caffe Mobilenet-SSD model for OpenVINO CPU AI threads...")
                OVstr = "SSDv1_cpu"
                OpenVINO_Thread.SSDv1 = True
                for i in range(nCPUthreads):
                    net.append(cv2.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD_deploy.prototxt", "MobileNetSSD/MobileNetSSD_deploy.caffemodel"))
                    net[i].setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
                    net[i].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            else:
                OpenVINO_Thread.SSDv1 = False
                if cv2.__version__ == "4.5.0-openvino" or cv2.__version__ == "4.5.1-openvino" or cv2.__version__ == "4.5.2-openvino":
                    print("[INFO] loading Tensor Flow Mobilenet-SSD v2 FP16 IR10 model for OpenVINO_2021.1...")
                    if useGPU:
                        OVstr = "SSDv2_IR10gpu"
                    else:
                        OVstr = "SSDv2_IR10cpu"
                else:
                    print("[INFO] loading Tensor Flow Mobilenet-SSD v2 FP32 model for OpenVINO CPU AI threads...")
                    OVstr = "SSDv2_FP32cpu"
                for i in range(nCPUthreads):
                    if cv2.__version__ == "4.5.0-openvino" or cv2.__version__ == "4.5.1-openvino" or cv2.__version__ == "4.5.2-openvino":
                        net.append(cv2.dnn.readNet("mobilenet_ssd_v2/MobilenetSSDv2cocoIR10.xml", "mobilenet_ssd_v2/MobilenetSSDv2cocoIR10.bin"))
                        net[i].setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
                        if useGPU:
                            print("Using OPEN_CL_FP16 GPU instead of CPU")
                            net[i].setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
                        else:
                            net[i].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    else:
                        net.append(cv2.dnn.readNet("mobilenet_ssd_v2/MobilenetSSDv2cocoFP32.xml", "mobilenet_ssd_v2/MobilenetSSDv2cocoFP32.bin"))
                        net[i].setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
                        net[i].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            import ocvdnn_CPU_Thread
            print("[INFO] loading Caffe Mobilenet-SSD model for ocvdnn CPU AI threads...")
            for i in range(nCPUthreads):
                net.append(cv2.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD_deploy.prototxt", "MobileNetSSD/MobileNetSSD_deploy.caffemodel"))
        # *** start CPU/GPU AI threads, usually only one makes sense, no attempt made to mix CPU and GPU threads
        # multiple CPU threads are allowed, but using GPU forces a single AI thread.
        CPUt = list()
        if cv2.__version__.find("openvino") > 0:
            if useGPU:
                print("[INFO] starting " + str(nCPUthreads) + " OpenVINO GPU AI Threads ...")
            else:
                print("[INFO] starting " + str(nCPUthreads) + " OpenVINO CPU AI Threads ...")
        else:
            print("[INFO] starting " + str(nCPUthreads) + " openCV dnn module CPU AI Threads ...")
        for i in range(nCPUthreads):
            if cv2.__version__.find("openvino") > 0:
                if yolo8_verify:
                    OpenVINO_Thread.__VERIFY_DIMS__ = (640,640)
                if yolo4_verify or OVyolo4_verify:
                    OpenVINO_Thread.__VERIFY_DIMS__ = (608,608)
                CPUt.append(Thread(target=OpenVINO_Thread.AI_thread,
                    args=(results, inframe, net[i], i, cameraLock, nextCamera, Ncameras,
                        PREPROCESS_DIMS, confidence-0.1, verifyConf-0.1, OVstr, blobThreshold, yoloQ)))
            else:
                if yolo8_verify:
                    ocvdnn_CPU_Thread.__VERIFY_DIMS__ = (640,640)
                if yolo4_verify or OVyolo4_verify:
                    ocvdnn_CPU_Thread.__VERIFY_DIMS__ = (608,608)
                CPUt.append(Thread(target=ocvdnn_CPU_Thread.AI_thread,
                    args=(results, inframe, net[i], i, cameraLock, nextCamera, Ncameras,
                        PREPROCESS_DIMS, confidence-0.1, verifyConf-0.1, "ocvdnnSSDv1", blobThreshold,  yoloQ)))
            CPUt[i].start()



    # *** Open second MQTT client thread for MQTTcam/# messages "MQTT cameras"
    # Requires rtsp2mqttDemand.py mqtt camera source
    # mqttCamsOneThread lets me try one mqtt thread for all MQTT cameras, need to re-evaluate after recent change to rtsp2mqttPdemand.py
    if Nmqtt > 0:
      mqttFrameDrops=[]
      mqttFrames=[]
      mqttCam=list()
      print("[INFO] connecting to " + MQTTcameraServer + " broker for MQTT cameras...")
      if not mqttCamsOneThread:   # use one MQTT thread per camera
        print("INFO starting one thread per MQTT camera.")
        j=0
        for i in camList:
            mqttFrameDrops.append(0)
            mqttFrames.append(0)
            mqttCam.append(mqtt.Client(userdata=(i, j), clean_session=True))
            mqttCam[j].on_connect = on_mqttCam_connect
            mqttCam[j].on_message = on_mqttCam
            mqttCam[j].on_publish = on_publish
            mqttCam[j].on_disconnect = on_disconnect
            mqttCam[j].connect(MQTTcameraServer, 1883, 60)
            mqttCam[j].loop_start()
            time.sleep(0.1)     # force thread dispatch
            if MQTTdemand:
                mqttCam[j].publish(str("sendOne/" + str(i)), "", 0, False)   # start messages
            j+=1
      else: # one MQTT thread for all cameras
        print("INFO all MQTT cameras will be handled in a single thread.")
        for i in camList:
            mqttFrameDrops.append(0)
            mqttFrames.append(0)
        mqttCam = mqtt.Client(userdata=camList, clean_session=True)
        mqttCam.on_connect = on_mqttCam_connect
        mqttCam.on_message = on_mqttCam
        mqttCam.on_publish = on_publish
        mqttCam.on_disconnect = on_disconnect
        mqttCam.connect(MQTTcameraServer, 1883, 60)
        mqttCam.loop_start()
        time.sleep(0.1)     # force thread dispatch
        if MQTTdemand:
            for i in camList:
                mqttCam.publish(str("sendOne/" + str(i)), "", 0, False)   # start messages



    if yolo4_verify:
        # Start darknet yolo v4 thread
        print("[INFO] Darknet yolo_v4 verification thread is starting ... ")
        yolo4=list()
        yolo4.append(Thread(target=darknet_Thread.yolov4_thread,args=(results, yoloQ)))
        yolo4[0].start()
        # wait for yolo thread to be running
        while darknet_Thread.__darknetThread__ is False:
            time.sleep(1.0)
        print("[INFO] Darknet yolo_v4 verification thread is running. ")

    if OVyolo4_verify:
        # Start darknet yolo v4 thread
        print("[INFO] OpenVINO yolo_v4 verification thread is starting ... ")
        yolo4ov=list()
        yolo4ov.append(Thread(target=yolo4OpenvinoVerification_Thread.yolo4ov_thread, args=(results, yoloQ)))
        yolo4ov[0].start()
        # wait for yolo thread to be running
        while yolo4OpenvinoVerification_Thread.__Thread__ is False:
            time.sleep(1.0)
        print("[INFO] OpenVINO yolo_v4 verification thread is running. ")

    if yolo8_verify:
        # Start Ultralytics yolo8 verification thread
        print("[INFO] Ultralytics yolo_v8 verification thread is starting ... ")
        yolo8=list()
        yolo8.append(Thread(target=yolo8_verification_Thread.yolov8_thread,args=(results, yoloQ)))
        yolo8[0].start()
        # wait for yolo thread to be running
        while yolo8_verification_Thread.__Thread__ is False:
            time.sleep(1.0)
        print("[INFO] Ultralytics yolo_v8 verification thread is running. ")

    #$$$# import yolo4 AI Thread
    if yolo4AI:
        import yolo4_AI_Thread
        # no pretense of multiple cuda AI threads
        y4AIthread=list()
        print("[INFO] Yolo_v4 AI thread is starting ... ")
        y4AIthread.append(Thread(target=yolo4_AI_Thread.AI_thread,
                args=(results, inframe, None, 0, cameraLock, nextCamera, Ncameras,
                    (608, 608), confidence, verifyConf-0.05, "Yolo4", blobThreshold, None)))    # lower verification confidence,  too many false negatives
        y4AIthread[0].start()
        while yolo4_AI_Thread.__Thread__ is False:
            time.sleep(1.0)
        print("[INFO] Yolo_v4 AI thread is running. ")

    if yolo8AI:
        import yolo8_AI_Thread
        # no pretense of multiple cuda AI threads
        yolo8_AI_Thread.__y8modelSTR__ = 'yolo8/yolov8x.pt'   # "best" , yolov8m.pt is "fastest",  yolov8l.pt is "in between"
        y8AIthread=list()
        print("[INFO] Yolo_v8 AI thread is starting ... ")
        YoloStr = "Yolo8" + yolo8_AI_Thread.__y8modelSTR__[12]
        y8AIthread.append(Thread(target=yolo8_AI_Thread.AI_thread,
                args=(results, inframe, None, 0, cameraLock, nextCamera, Ncameras,
                    (640, 640), confidence, verifyConf-0.05, YoloStr, blobThreshold, None)))    # lower verification confidence,  too many false negatives
        y8AIthread[0].start()
        while yolo8_AI_Thread.__Thread__ is False:
            time.sleep(1.0)
        print("[INFO] Yolo_v8 AI thread is running. ")



    # starting rtsp threads can take a long time, send image and message to dasboard to indicate progress
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (127,127,192)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    client.publish("ImageBuffer/!Starting RTSP threads, this can take awhile.", bytearray(img_as_jpg), 0, False)

    # *** start camera reading threads
    ### Try moving camera threads start up until after verification thread started
    o = list()
    if Nonvif > 0:
        import onvif_Thread
        print("[INFO] starting " + str(Nonvif) + " Onvif Camera Threads ...")
        for i in range(Nonvif):
            onvif_Thread.__CamName__ = CamName
            o.append(Thread(target=onvif_Thread.onvif_thread, args=(inframe[i], i, CameraURL[i])))
            o[i].start()

    if Nrtsp+Nfisheye > 0:
        global threadLock
        global threadsRunning
        threadLock = Lock()
        threadsRunning = 0
        for i in range(Nrtsp):
            rtsp_thread.__CamName__ = CamName
            o.append(Thread(target=rtsp_thread, args=(inframe[i+Nonvif], i+Nonvif, rtspURL[i])))
            o[i+Nonvif].start()
        FEoffset=FishEyeOffset
        for i in range(Nfisheye):
            Nfe=len(PTZparam[i])-1  # first entry is camera resolution, not PTZ view parameters
##            print(PTZparam[i])
            ### def FErtsp_thread(inframe, Nfe, FEoffset, PTZparam, camn, URL):
            o.append(Thread(target=FErtsp_thread, args=(inframe, Nfe, FEoffset, PTZparam[i], FEoffset+i, FErtspURL[i])))  # for virtual camera
            o[i+Nonvif+Nrtsp].start()
            FEoffset+=Nfe
        # make sure rtsp threads are all running
        while threadsRunning < Nrtsp+Nfisheye:
            time.sleep(0.5)
        print("[INFO] All " + str(Nrtsp+Nfisheye) + " RTSP Camera Sampling Threads are running.")



    #*************************************************************************************************************************************
    # *** enter main program loop (main thread)
    # loop over frames from the camera and display results from AI_thread
    excount=0
    aliveCount=0
    SEND_ALIVE=100  # send MQTT message approx. every SEND_ALIVE/fps seconds to reset external "watchdog" timer for auto reboot.
    waitCnt=0
    detectCount=0
    prevUImode=UImode
    currentDT = datetime.datetime.now()
    client.publish("AI/Status", "Python AI running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)
    # *** MQTT send a blank image to the dashboard UI
    print("[INFO] Clearing dashboard ...")
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (127,192,127)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    client.publish("ImageBuffer/!AI has Started.", bytearray(img_as_jpg), 0, False)
    #start the FPS counter
    print("[INFO] starting the FPS counter ...")
    fps = FPS().start()
    print("[INFO] AI/Status: Python AI running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))

    while not QUIT:
        try:
            try:
                (img, cami, personDetected, dt, ai, bp, yolo_frame) = results.get(True,0.100)  # perhaps yolo_frame should be zoom_frame instead
            except Exception as e:
                #print(e)
                waitCnt+=1
                img=None
                aliveCount = (aliveCount+1) % SEND_ALIVE   # MQTTcam images stop while Lorex reboots, recovers eventually so keep alive
                if aliveCount == 0:
                    client.publish("AmAlive", "true", 0, False)
                ##cv2.waitKey(1)
                continue
            if img is not None:
                fps.update()    # update the FPS counter
                # setup for file saving
                folder=dt.strftime("%Y-%m-%d")
                filename=dt.strftime("%H_%M_%S.%f")
                filename=filename[:-4] + "_" + ai  #just keep tenths, append AI source
                # setup for local save of yolo frame of zoomed image of detection
                # currently detection images saved by node-red if -ls option not active, I'm currently rethinking this
                yfolder=str(detectPath + "/" + folder)
                if os.path.exists(yfolder) == False:
                    os.mkdir(yfolder)
                    if not __DEBUG__ and not NoSaveZoom:
                        if os.path.exists(str(yfolder + "/zoom")) == False:
                            os.mkdir(str(yfolder + "/zoom"))     # put detection zoom into sub-folder
                #''' Debug code to see verification failure images
                if __DEBUG__:
                    ##if (yolo4_verify or yolo8_verify) and yolo_frame is not None:
                    if yolo_frame is not None:
                        if personDetected:
                            ##outName=str(yfolder + "/" + filename + "_" + "zoom_Cam" + str(cami) +"_AI.jpg")
                            outName=str(yfolder + "/zoom/" + filename + "_zoom_" + CamName[cami] +"_AI.jpg")
                        else:
                            if bp[0] == -1: # failed AI zoom redetection
                                ##outName=str(yfolder + "/" + filename + "_" + "ZnoV_Cam" + str(cami) +".jpg")
                                outName=str(yfolder + "/" + filename + "_ZnoV_" + CamName[cami] +".jpg")
                            if bp[0] == -2: # failed Yolo zoom detection
                                ##outName=str(yfolder + "/" + filename + "_" + "YnoV_Cam" + str(cami) +".jpg")
                                outName=str(yfolder + "/" + filename + "_YnoV_" + CamName[cami] +".jpg")
                        cv2.imwrite(outName, yolo_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                #'''
                if personDetected:  # personDetected implies yolo_frame is not None
                    detectCount+=1
                    if CLAHE:   # create CLAHE frame
                        if nodeRedFull:
                            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        else:
                            lab = cv2.cvtColor(yolo_frame, cv2.COLOR_BGR2LAB)
                        lab_planes = cv2.split(lab)
                        lab_planes[0] = clahe.apply(lab_planes[0])
                        lab = cv2.merge(lab_planes)
                        CLAHE_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                        retv, img_as_jpg = cv2.imencode('.jpg', CLAHE_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])    # write clahe image to node-red
                    else:
                        if nodeRedFull:
                            retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])        # write full frame to node-red controller
                        else:
                            retv, img_as_jpg = cv2.imencode('.jpg', yolo_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])    # write zoomed image to node-red
                    if retv:
                        if nodeRedFull:
                            ##outName=str("AIdetection/!detect/" + folder + "/" + filename + "_" + "Full_Cam" + str(cami) +"_AI.jpg")
                            outName=str("AIdetection/!detect/" + folder + "/alert/" + filename + "_" + CamName[cami] +"_AI.jpg")
                        else:
                            ##outName=str("AIdetection/!detect/" + folder + "/" + filename + "_" + "Zoom_Cam" + str(cami) +"_AI.jpg")
                            outName=str("AIdetection/!detect/" + folder + "/alert/" + filename + "_Zoom_" + CamName[cami] +"_AI.jpg")
                        outName=outName + "!" + str(bp[0]) + "!" + str(bp[1]) + "!" + str(bp[2]) + "!" + str(bp[3]) + "!" + str(bp[4]) + "!" + str(bp[5]) + "!" + str(bp[6]) + "!" + str(bp[7])
                        client.publish(str(outName), bytearray(img_as_jpg), 0, False)
                        ##print(outName)  # log detections
                        if not NoLocalSave or __DEBUG__:
                            # save all AI person detections and zoom image no matter the ALARM_MODE, may change this later to not save in IDLE mode.
                            # part of Debug code to see yolo verification images
                            if not __DEBUG__ and not NoSaveZoom:
                                ##outName=str(yfolder + "/" + filename + "_" + "zoom_Cam" + str(cami) +"_AI.jpg")
                                outName=str(yfolder + "/zoom/" + filename + "_zoom_" + CamName[cami] +"_AI.jpg")
                                cv2.imwrite(outName, yolo_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])  # yolo frame is zoom frame if not y4v or y8v option
                            ##outName=str(yfolder + "/" + filename + "_" + "full_Cam" + str(cami) +"_AI.jpg")
                            outName=str(yfolder + "/" + filename + "_" + CamName[cami] +"_AI.jpg")
                            cv2.imwrite(outName, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                    else:
                        print("[INFO] conversion of np array to jpg in buffer failed!")
                        continue
                # send image for live display in dashboard
                if ((CameraToView == cami) and (UImode == 1 or (UImode == 2 and personDetected))) or (UImode ==3 and personDetected):
                    if personDetected:
                        ##topic=str("ImageBuffer/!" + filename + "_" + "Cam" + str(cami) +"_AI.jpg")
                        topic=str("ImageBuffer/!" + filename + "_" + CamName[cami] +"_AI.jpg")
                    else:
                        retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                        if retv:
                            ##topic=str("ImageBuffer/!" + filename + "_" + "Cam" + str(cami) +".jpg")
                            topic=str("ImageBuffer/!" + filename + "_" + CamName[cami] +".jpg")
                        else:
                            print("[INFO] conversion of numpy array to jpg in buffer failed!")
                            continue
                    client.publish(str(topic), bytearray(img_as_jpg), 0, False)
                # display the frame to the screen if enabled, in normal usage display is 0 (off)
                if dispMode > 0:
                    name=str("Live_" + CamName[cami])
                    cv2.imshow(name, cv2.resize(img, (imwinWidth, imwinHeight)))
                    key = cv2.waitKey(1) ###& 0xFF
                    ###if key == ord("q"): # if the `q` key was pressed, break from the loop
                    ###    QUIT=True   # exit main loop
                    if (yolo4_verify or yolo8_verify) and yolo_frame is not None:
                        if personDetected:
                            cv2.imshow("yolo_verify", yolo_frame)
                        else:
                            cv2.imshow("yolo_reject", yolo_frame)
                        key = cv2.waitKey(1) ### & 0xFF
                        ###if key == ord("q"): # if the `q` key was pressed, break from the loop
                        ###    QUIT=True   # exit main loop
                    else:
                        if personDetected:
                            cv2.imshow("detection_zoom", yolo_frame)
                            cv2.waitKey(1)
                        ###key = cv2.waitKey(1) & 0xFF
                        ###if key == ord("q"): # if the `q` key was pressed, break from the loop
                        ###    QUIT=True   # exit main loop
                else:
                    if show_zoom:
                        if personDetected:
                            cv2.imshow("detection_zoom", yolo_frame)
                            cv2.waitKey(1)

                aliveCount = (aliveCount+1) % SEND_ALIVE
                if aliveCount == 0:
                    client.publish("AmAlive", "true", 0, False)
                    cv2.waitKey(1)      # try to keep detection_zoom window display alive
                if prevUImode != UImode:
                    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
                    img[:,:] = (154,127,100)
                    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                    client.publish("ImageBuffer/!AI Mode Changed.", bytearray(img_as_jpg), 0, False)
                    prevUImode=UImode
            ##else:   # img is None
            ##    cv2.waitKey(1)
        # if "ctrl+c" is pressed in the terminal, break from the loop
        except KeyboardInterrupt:
            QUIT=True   # exit main loop
            ##continue
        except Exception as e:
            currentDT = datetime.datetime.now()
            print(" **** Main Loop Error: " + str(e)  + currentDT.strftime(" -- %Y-%m-%d %H:%M:%S.%f"))
            excount=excount+1
            if excount <= 3:
                continue    # hope for the best!
            else:
                break       # give up! Hope watchdog gets us going again!
    #end of while not QUIT  loop
    #*************************************************************************************************************************************

    # *** Clean up for program exit
    fps.stop()    # stop the FPS counter timer
    currentDT = datetime.datetime.now()
    print("\n[INFO] Program Exit signal received:" + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    # display FPS information
    print("*** AI processing approx. FPS: {:.2f} ***".format(fps.fps()))
    print("[INFO] Run elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Frames processed by AI system: " + str(fps._numFrames))
    print("[INFO] Person Detection by AI system: " + str(detectCount))
    print("[INFO] Main loop waited for results: " + str(waitCnt) + " times.")
    currentDT = datetime.datetime.now()
    client.publish("AI/Status", "Python AI stopped." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)
    print("[INFO] AI/Status: Python AI stopped." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))

    # Send a blank image the dashboard UI
    print("[INFO] Clearing dashboard ...")
    img = np.zeros((imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (32,32,32)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    client.publish("ImageBuffer/!AI has Exited.", bytearray(img_as_jpg), 0, False)
    time.sleep(1.0)

    # stop Yolo v4 AI Thread
    # there seem to be threading issues between python and darknet C code I don't fully understand
    #I need to stop the darknet yolo4 thread before anything else to avoid a segfault on exit
    if yolo4AI:
        print("[INFO] Stopping Yolo4 AI Thread ...")
        yolo4_AI_Thread.__Thread__ = False
        y4AIthread[0].join()
        print("[INFO] Yolo4 AI Thread has exited.")

    if yolo4_verify:
        print("[INFO] Stopping yolo4 verification Thread ...")
        darknet_Thread.__darknetThread__ = False
        yolo4[0].join()
        print("[INFO] yolov4 verification Thread has exited.")

    if OVyolo4_verify:
        print("[INFO] Stopping OpenVINO yolo4 verification Thread ...")
        yolo4OpenvinoVerification_Thread.__Thread__ = False
        yolo4ov[0].join()
        print("[INFO] OpenVINO yolov4 verification Thread has exited.")

    # stop MQTT cameras
    if Nmqtt > 0:
      if not mqttCamsOneThread:
        for i in range(Nmqtt):
            mqttCam[i].disconnect()
            mqttCam[i].loop_stop()
            print("MQTTcam/" + str(camList[i]) + " has dropped: " + str(mqttFrameDrops[i]) + " frames out of: " + str(mqttFrames[i]))
      else:
        mqttCam.disconnect()
        mqttCam.loop_stop()
        for i in range(Nmqtt):
            print("MQTTcam/" + str(camList[i]) + " has dropped: " + str(mqttFrameDrops[i]) + " frames out of: " + str(mqttFrames[i]))

    # Stop and wait for capture threads to exit
    if Nonvif > 0:
        print("[INFO] Stopping Onvif camera threads ...")
        onvif_Thread.__onvifThread__ = False
        for i in range(Nonvif):
            o[i].join()
        print("[INFO] All Onvif camera threads have exited.")
    if Nrtsp > 0:
        print("[INFO] Stopping RTSP camera threads ...")
        __rtspThread__ = False
        for i in range(Nrtsp):
            o[i+Nonvif].join()
        print("[INFO] All RTSP camera threads have exited.")
    if Nfisheye > 0:
        print("[INFO] Stopping Fisheye camera threads ...")
        __fisheyeThread__ = False
        for i in range(Nfisheye):
            o[i+Nonvif+Nrtsp].join()
        print("[INFO] All Fisheye camera threads have exited.")

    # stop AI threads
    if nCoral > 0:
        print("[INFO] Stopping TPU Thread ...")
        Coral_TPU_Thread.__Thread__ = False   # maybe my QUITf() was clenaer, but I can stage the thread exits for debugging.
        #$$$# Prototype_AI_Thread.__protoThread__ = False
        for i in range(nCoral):
            Ct[i].join()
        print("[INFO] All Coral TPU AI Threads have exited.")

    # Stop OpenVINO CPU and NCS threads
    if nCPUthreads > 0:
        print("[INFO] Stopping CPU AI  Threads ...")
        if cv2.__version__.find("openvino") > 0:
            OpenVINO_Thread.__Thread__ = False
        else:
            ocvdnn_CPU_Thread.__Thread__ =False
        for i in range(nCPUthreads):
            CPUt[i].join()
        print("[INFO] All CPU AI Threads have exited.")
    if nNCSthreads > 0:
        print("[INFO] Stopping NCS2 Threads ...")
        OpenVINO_Thread.__Thread__ = False
        for i in range(nNCSthreads):
            OVt[i].join()
        print("[INFO] All OpenVINO Myriad NCS2 AI Threads have exited,")

    #$$$#  stop yolo verify thread
    if yolo8_verify:
        print("[INFO] Stopping yolo8 verification Thread ...")
        yolo8_verification_Thread.__Thread__ = False
        yolo8[0].join()
        print("[INFO] yolov8 verification Thread has exited.")

    # stop Yolo v8 AI Thread
    if yolo8AI:
        print("[INFO] Stopping Yolo8 AI Thread ...")
        yolo8_AI_Thread.__Thread__ = False
        y8AIthread[0].join()
        print("[INFO] Yolo8 AI Thread has exited.")

    # destroy all windows if we are displaying them
    if args["display"] > 0:
        cv2.destroyAllWindows()



    # clean up MQTT
    client.disconnect()     # normal exit, Will message should not be sent.
    currentDT = datetime.datetime.now()
    print("[INFO] Stopping MQTT Threads." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    client.loop_stop()      ### Stop MQTT thread


    # bye-bye
    currentDT = datetime.datetime.now()
    print("Program Exit." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    print("$$$**************************************************************$$$")
    print("")
    print("")




# *** RTSP Sampling Thread
#******************************************************************************************************************
# rtsp stream sampling thread
### 20JUN2019 wbk much improved error handling, can now unplug & replug a camera, and the thread recovers
def rtsp_thread(inframe, camn, URL):
    global threadLock
    global threadsRunning
    global __rtspThread__
    global __CamName__
    global CamName
    
    __rtspThread__ = True
    __CamName__ = CamName
    ocnt=0
    Error=False
    Error2=False
    currentDT = datetime.datetime.now()
    print("[INFO] RTSP stream sampling thread " + __CamName__[camn] + " is starting..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    Rcap=cv2.VideoCapture(URL)
    Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)     # doesn't throw error or warning in python3, but not sure it is actually honored
    threadLock.acquire()
    currentDT = datetime.datetime.now()
    print("[INFO] RTSP stream sampling thread " + __CamName__[camn] + " is running..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    threadsRunning += 1
    threadLock.release()
    while __rtspThread__ is True:
         # grab the frame
        try:
            if Rcap.isOpened() and Rcap.grab():
                gotFrame, frame = Rcap.retrieve()
            else:
                frame = None
                if not Error:
                    Error=True
                    currentDT = datetime.datetime.now()
                    print('[Error!] RTSP Camera '+ __CamName__[camn] + ': ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                          '\n    ' + URL[0:38] + '\n        Will close and re-open Camera ' + __CamName__[camn] +' RTSP stream in attempt to recover.')
                # try closing the stream and reopeing it, I have one straight from China that does this error regularly
                # NOTE this only detects rstp connection loss, if the DVR sends black frames on camera loss this will not detect it!
                Rcap.release()
                if not Error2:
                    time.sleep(60.0)     # does this help or hurt or no difference? Always Seems to take a minute of more to recover
                Rcap=cv2.VideoCapture(URL)
                Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if not Rcap.isOpened() :
                    if not Error2:
                        Error2=True
                        currentDT = datetime.datetime.now()
                        ##print('   [Error2!] RTSP stream'+ str(camn) + ' re-open failed!' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                        ##     '\n   Will loop closing and re-opening Camera' + str(camn) +' RTSP stream, further messages suppressed.')
                        print('   [Error2!] RTSP stream '+ __CamName__[camn] + ' re-open failed!' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                              '\n   Will loop closing and re-opening Camera ' + __CamName__[camn] +' RTSP stream, further messages suppressed.')
                    time.sleep(30.0)
                continue
            if gotFrame: # path for sucessful frame grab, following test is in case error recovery is in progress
                if Error:   # log when it recovers
                    currentDT = datetime.datetime.now()
                    print('[$$$$$$] RTSP Camera '+ __CamName__[camn] + ' has recovered: --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                          '\n    ' + URL[0:38] + "\n")
                    Error=False    # after geting a good frame, enable logging of next error
                    Error2=False
        except Exception as e:
            frame = None
            currentDT = datetime.datetime.now()
            print('[Exception] RTSP stream '+ __CamName__[camn] + ': ' + str(e) + '\n    ' + URL[0:38] + ' :  ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
            time.sleep(10.0)
        try:
            if frame is not None:
                imageDT=datetime.datetime.now()
                if inframe.full():
                    [_,_,_]=inframe.get(False)    # remove oldest sample to make space in queue
                    ocnt+=1     # if happens here shouldn't happen below
                inframe.put((frame.copy(), camn, imageDT), False)   # no block if queue full, go grab fresher frame
        except: # most likely queue is full, Python queue.full() is not 100% reliable
            # a large drop count for rtsp streams is not a bad thing as we are trying to keep the input buffers nearly empty to reduce latency.
            ocnt+=1
    Rcap.release()
    print("RTSP stream sampling thread " + __CamName__[camn] + " is exiting, dropped frames " + str(ocnt) + " times.")




## Fisheye Window snippet
# --> https://github.com/daisukelab/fisheye_window
class FishEyeWindow(object):
    """Fisheye Window class
    You can get image out of your fisheye image for desired view.
    1. Create instance by feeding image sizes.
    2. Call buildMap to set the view you want.
       This calculates the map for the 'remap.'
    3. Call getImage that simply remaps.
    """
    def __init__(
            self,
            srcWidth,
            srcHeight,
            destWidth,
            destHeight
        ):
        # Initial parameters
        self._srcW = srcWidth
        self._srcH = srcHeight
        self._destW = destWidth
        self._destH = destHeight
        self._al = 0
        self._be = 0
        self._th = 0
        self._R  = srcWidth / 2
        self._zoom = 1.0
        # Map storage
        self._mapX = np.zeros((self._destH, self._destW), np.float32)
        self._mapY = np.zeros((self._destH, self._destW), np.float32)
    def buildMap(self, alpha=None, beta=None, theta=None, R=None, zoom=None):
        # Set the angle parameters
        self._al = (alpha, self._al)[alpha == None]
        self._be = (beta, self._be)[beta == None]
        self._th = (theta, self._th)[theta == None]
        self._R = (R, self._R)[R == None]
        self._zoom = (zoom, self._zoom)[zoom == None]
        # Build the fisheye mapping
        al = self._al / 180.0
        be = self._be / 180.0
        th = self._th / 180.0
        A = np.cos(th) * np.cos(al) - np.sin(th) * np.sin(al) * np.cos(be)
        B = np.sin(th) * np.cos(al) + np.cos(th) * np.sin(al) * np.cos(be)
        C = np.cos(th) * np.sin(al) + np.sin(th) * np.cos(al) * np.cos(be)
        D = np.sin(th) * np.sin(al) - np.cos(th) * np.cos(al) * np.cos(be)
        mR = self._zoom * self._R
        mR2 = mR * mR
        mRsinBesinAl = mR * np.sin(be) * np.sin(al)
        mRsinBecosAl = mR * np.sin(be) * np.cos(al)
        centerV = int(self._destH / 2.0)
        centerU = int(self._destW / 2.0)
        centerY = int(self._srcH / 2.0)
        centerX = int(self._srcW / 2.0)
        # Fill in the map, slows dramatically with large view (destination) windows
        for absV in range(0, int(self._destH)):
            v = absV - centerV
            vv = v * v
            for absU in range(0, int(self._destW)):
                u = absU - centerU
                uu = u * u
                upperX = self._R * (u * A - v * B + mRsinBesinAl)
                lowerX = np.sqrt(uu + vv + mR2)
                upperY = self._R * (u * C - v * D - mRsinBecosAl)
                lowerY = lowerX
                x = upperX / lowerX + centerX
                y = upperY / lowerY + centerY
                _v = (v + centerV, v)[centerV <= v]
                _u = (u + centerU, u)[centerU <= u]
                self._mapX.itemset((_v, _u), x)
                self._mapY.itemset((_v, _u), y)

    def getImage(self, img):
        # Look through the window
        output = cv2.remap(img, self._mapX, self._mapY, cv2.INTER_LINEAR)
        #output = cv2.remap(img, self._mapX, self._mapY, cv2.INTER_CUBIC) # no significant improvement on 4 Mpixel test image
        return output


# create virtual cameras from PTZ crops from a fisheye camera rtsp stream
# Note the PTZ param are string variables read from the fisheye.rtsp text file
# created with the interactive fisheye_window C++ utility program.
def FErtsp_thread(inframe, Nfe, FEoffset, PTZparam, camn, URL):
    global __fisheyeThread__
    global threadLock
    global threadsRunning
    ocnt=[]
    for i in range(Nfe):
        ocnt.append(0)      # init counter array
    fe=[]
    Error=False
    Error2=False
##    print(PTZparam)
    threadLock.acquire()
    mapFilename="fisheye" +str(camn)+ "_map"
    try:
      filehandler = open(mapFilename, 'rb')
      currentDT = datetime.datetime.now()
      print( "Loading " + mapFilename + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
      fe = pickle.load(filehandler)
      filehandler.close()
    except:
      currentDT = datetime.datetime.now()
      print( "Creating " + mapFilename + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
      for i in range(Nfe):
        if i == 0:
            srcW=int(PTZparam[0][0])
            srcH=int(PTZparam[0][1])
        # PTZparam = [ [srcW,srcH], [destW, destH, alpha, beta, theta, zoom], [...] ] chosen with fisheye_window  C++ utility
        print("FE" +str(camn)+ " PTZview" +str(i)+ " " +str(PTZparam[i+1]))
        fe.append(FishEyeWindow(srcW, srcH, int(PTZparam[i+1][0]), int(PTZparam[i+1][1])))    # instance a view with desired output image size
        fe[i].buildMap(alpha=float(PTZparam[i+1][2]), beta=float(PTZparam[i+1][3]),
                       theta=float(PTZparam[i+1][4]), zoom=float(PTZparam[i+1][5]))    # build map for this PTZ view
      currentDT = datetime.datetime.now()
      print("Saving FE" +str(camn)+ " virtual PTZ views as: " + mapFilename + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
      filehandler = open(mapFilename, 'wb')
      pickle.dump(fe, filehandler)
      filehandler.close()
    currentDT = datetime.datetime.now()
    print("[INFO] Fisheye Camera RTSP stream FE" + str(camn) + " is opening..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    Rcap=cv2.VideoCapture(URL)
    Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)     # doesn't throw error or warning in python3, but not sure it is actually honored
#    threadLock.acquire()
    currentDT = datetime.datetime.now()
    print("[INFO] Fisheye RTSP stream sampling thread" + str(camn) + " is running..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    threadsRunning += 1
    threadLock.release()
    while __fisheyeThread__ is True:
         # grab the frame
        try:
            if Rcap.isOpened() and Rcap.grab():
                gotFrame, frame = Rcap.retrieve()
            else:
                frame = None
                if not Error:
                    Error=True
                    currentDT = datetime.datetime.now()
                    print('[Error!] RTSP Camera'+ str(camn) + ': ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                          '\n   ' + URL[0:38] + '\n   Will close and re-open Camera' + str(camn) +' RTSP stream in attempt to recover.')
                # try closing the stream and reopeing it, I have one straight from China that does this error regularly
                # NOTE this only detects rstp connection loss, if the DVR sends black frames on camera loss this will not detect it!
                Rcap.release()
                time.sleep(5.0)     # does this help or hurt?
                Rcap=cv2.VideoCapture(URL)
                Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if not Rcap.isOpened() :
                    if not Error2:
                        Error2=True
                        currentDT = datetime.datetime.now()
                        print('[Error2!] RTSP stream'+ str(camn) + ' re-open failed! $$$  --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                              '\n   Will loop closing and re-opening Camera' + str(camn) +' RTSP stream, further messages suppressed.')
                    time.sleep(10.0)
                continue
            if gotFrame: # path for sucessful frame grab, following test is in case error recovery is in progress
                if Error:   # log when it recovers
                    currentDT = datetime.datetime.now()
                    print('[$$$$$$] RTSP Camera'+ str(camn) + ' has recovered: --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") +
                          '\n   ' + URL[0:38] + "\n")
                    Error=False    # after geting a good frame, enable logging of next error
                    Error2=False
        except Exception as e:
            frame = None
            currentDT = datetime.datetime.now()
            print('[Exception] RTSP stream'+ str(camn) + ': ' + str(e) + '\n ' + URL[0:38] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
            time.sleep(10.0)
        if frame is not None:
            imageDT = datetime.datetime.now()
            for i in range(Nfe):
                try:
                    if inframe[FEoffset+i].full():
                        [_,_,_]=inframe[FEoffset+i].get(False)    # remove oldest sample to make space in queue
                        ocnt[i]+=1   # it this happens here, it shouldn't happen below
                    PTZview=fe[i].getImage(frame)
                    inframe[FEoffset+i].put((PTZview.copy(), FEoffset+i, imageDT), True)  ## force this frame to complete in all queues
                except: # most likely queue is full, Python queue.full() is not 100% reliable
                    # a large drop count for rtsp streams is not a bad thing as we are trying to keep the input buffers nearly empty to reduce latency.
                    ocnt[i]+=1

    Rcap.release()
    print("RTSP Fisheye Camera sampling thread" + str(camn) + " is exiting ...")
    for i in range(Nfe):
        print("   Fisheye Cam "+ str(FEoffset+i) +" dropped frames " + str(ocnt[i]) + " times.")






# python boilerplate
if __name__ == '__main__':
    main()



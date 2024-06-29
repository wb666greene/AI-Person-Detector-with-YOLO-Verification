import numpy as np
import cv2
import requests
import time
import datetime
from PIL import Image
from io import BytesIO

global __onvifThread__
__onvifThread__ = False
global __CamName__


## *** ONVIF Sampling Thread ***
#******************************************************************************************************************
# Onvif camera sampling thread
def onvif_thread(inframe, camn, URL):
    global __onvifThread__
    global __CamName__

    print("[INFO] ONVIF Camera " + __CamName__[camn] + " thread is running...")
    ocnt=0  # count of times inframe thread output queue was full (dropped frames)
    Error=0
    __onvifThread__ = True
    while __onvifThread__ is True:
        # grab the frame
        try:
            r = requests.get(URL, timeout=1.0)
            i = Image.open(BytesIO(r.content))
            frame = np.array(i)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if Error >= 4 and frame is not None:   # log when it recovers
                currentDT = datetime.datetime.now()
                print('[******] Onvif cam '+ __CamName__[camn] + ': recovered: ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") + ' ' + URL)
                Error=0    # after getting a good frame, enable logging of next error
        except Exception as e:
            # this appears to fix the Besder camera problem where it drops out for minutes every 5-12 hours
            Error+=1
            if Error == 4:   # suppress the zillions of sequential error messages while it recovers
                currentDT = datetime.datetime.now()
                ## printing the error string hasn't been particularly informative
                print('[Error!] Onvif cam '+ __CamName__[camn] + ': ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S") + ' ' + URL)
                ##print('[Error!] Onvif cam'+ str(camn) + ': ' +  URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
            frame = None
            time.sleep(1.0)     # let other threads have more time while this camera recovers, which sometimes takes minutes
        try:
            if frame is not None and __onvifThread__ is True:
                imageDT = datetime.datetime.now()
                inframe.put((frame, camn, imageDT), True, 0.200)
                Error=0     # reset error count when good frame is received.
                ##time.sleep(sleepyTime)   # force thread switch, hopefully smoother sampling, 10Hz seems upper limit for snapshots
        except: # most likely queue is full
            if __onvifThread__ is False:
                break
            ocnt=ocnt+1
            ##time.sleep(sleepyTime)
            continue
    print("ONVIF Camera " + __CamName__[camn] + " thread is exiting, dropped frames " + str(ocnt) + " times.")



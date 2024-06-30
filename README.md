# AI-Person-Detector-with-YOLO-Verification
This is the next evolution of https://github.com/wb666greene/AI-Person-Detector
The major change was ading an intermideate output queue to verify detections with a YOLO model and removing the Raspberry Pi support (where YOLO is not viable at present).

Here is an image sequence that shows the software in action, made from the detection images as a solicitor walks from my neighbor's yard, across the street, to my front door, leaves his flyer, and proceeds across my yard on to the next house. He walks across the field of view of multiple cameras with various resolutions of 4K, 5Mpixel, & 1080p:
https://youtu.be/XZYyE_WsRLI
![07_03_48 82_TPU_CliffwoodSouth_AI](https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification/assets/31488806/26362151-1808-46cc-90b5-ca85973f2a60)

&nbsp;

I've had some health issues that laid me up for about a year, but this system has been running 24/7/365 since about September of 2022 with 26 outside cameras each set for 3 fps and has had only four false positives in all that time. Two of the false positives were my neighbor's cat mis-detected as a person that also detected as a person with the YOLO model:
![01_51_24 97_TPU_driveway_AI](https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification/assets/31488806/d67224a1-332f-4e8a-a6cf-611fe2935d15)
![22_02_20 66_TPU_SideYard_AI](https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification/assets/31488806/5f18a4a9-ec20-4220-975d-b801b1cbb042)
and two false positives from out of focus blobs of bugs too close to the camera.  I have a filter in node-red on these cameras to reject detections that are too small or too large to be a real person in the field of view that has stopped most of the bug blob false alerts but not these two:
![01_07_30 57_TPU_CliffwoodDoor_AI](https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification/assets/31488806/25a308d8-0450-4068-9949-0a79cadaf1ae)
![01_16_00 24_TPU_driveway_AI](https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification/assets/31488806/34ff6270-5940-483f-bb60-e989d9b98636)

&nbsp;

Basic operation is that a thread is launched for each RTSP or ONVIF camera. The MobilenetSSD_v2 AI thread reads the queues round-robin, resizes the frame to 300x300 and if a person is detected a digital zoom is done on the detection which is resized to 300x300 and the inference repeated.  If it exceeds the detection threshold the zoomed image is resized for the YOLO model, currently Darknet YOLO4 ( https://github.com/AlexeyAB/darknet ), Darknet YOLO4 OpenVINO ( https://github.com/TNTWEN/OpenVINO-YOLOV4 ) or Ultralytics YOLO8 ( https://docs.ultralytics.com/ ) and written to an intermediate output queue that does the YOLO inference.  If this detection passes the "person detection" threshold it is written to the final output queue that is read by the main program and passed to the node-red system for "electric fence" and other ad-hoc validity filtering before sending audio and/or MMS and Email alerts depending on the alarm system mode.  All detections are saved independent of alarm system mode and the node-red runs a daily process to remove images older than 30 days.

&nbsp;

Ultralytics YOLO8 is the easiest to use if you have an Nvida GPU and CUDA installed.  Basic setup instructions are in the files: OV_LTS_Ubuntu22.04_setup.txt, Ubuntu22.04_setup_notes.txt, & ubuntu20.04_AI_setup.txt.  OpenVINO is very dynamic and the Movidius NCS2 support has been removed from newer versions, only OpenVINO 2021.3 oe 2021.4 work for the yolo4OpenvinoVerification_Thread.py.  OpenVINO verification works quite well with TPU or CPU MobilenetSSD_v2 initial detection and Intel GPU yolo4 verification -- a bottom of the line Lenovo i3 11th generation laptop ($200 closeout as "last years model) with integrated GPU and CPU initial detection suports four 4K cameras nicely.  The NCS2 can only do about 2fps for the YOLO verification but it has worked well for me on an i3 system with TPU to support five 720p and one 1080p ONVIF cameras along with one 5 Mpixel RTSP camera.

&nbsp;

Unfortunately the models are too big to upload to github (and some may not be allowed), so you have to download them from the original source and convert to the OpenVINO format, if necessary, yourself.  Raise an issue if you need help.

If you want to use __MobilenetSSD_v1__ I managed to upload the model to the previous version at: https://github.com/wb666greene/AI-Person-Detector you'll need MobileNetSSD/MobileNetSSD_deploy.prototxt, 
 & MobileNetSSD/MobileNetSSD_deploy.caffemodel.
 
For __TPU Mobilenet_SSD_v2__ you'll need mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite & coco_labels.txt in a mobilenet_ssd_v2/ folder from: https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite and https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt

For __Ultralytics__ YOLO8 I believe the code is automatically downloaded if it is not where the __y8modelSTR__ specifies: 'yolo8/yolov8x.pt', you must also have CUDA installed.

For __Darknet__ YOLO4 you need to have CUDA installed and build the libdarknet.so library yourself (system specific) instructions at: https://github.com/AlexeyAB/darknet

For __OpenVINO YOLO4__, follow the instructions at: https://github.com/TNTWEN/OpenVINO-YOLOV4 be aware that only OpenVINO 2021.3 and 2021.4 will work, you need to end up with frozen_darknet_yolov4_model.xml & frozen_darknet_yolov4_model.bin  in a yolo4_OpenVINO/fp16/ folder.

For CPU initial detection you'll also need cv2 version "4.5.0-openvino" or "4.5.1-openvino" or "4.5.2-openvino" with MobilenetSSDv2cocoIR10.xml & MobilenetSSDv2cocoIR10.bin in a  mobilenet_ssd_v2/ folder. It has been a real long time since I did these conversion, if you have trouble, raise an isse and I'll try to help.

&nbsp;

Ultralytics has recently released a YOLO10 model and OpenVINO support, I'm investigating this now, my next project will be to stop trying to support all the old stuff and simplify the installation and setup to only support CPU & TPU for MobilenetSSD_v2 initial detection and ultralytics YOLO for the verification, I'm basically waiting for the nest LTS version of OpenVINO.

&nbsp;

If you are comfortable with Docker and especially if you also run Home Assistant, take a look at Frigate: https://docs.frigate.video/ he has put tremendous effort into traditional NVR features and User Interface and has some YOLO support, my design goal was "set and forget" with only minumal user interface done in node-red, normally we only interact with it via the audio alerts when we are at home and via MMS and Email messages when we are away.

&nbsp;

Here is the minimal User Interface done with node-red and viewd in Chromium after it detected a couple walking in the street, this did not triger an alert since they were outside the 'electric fence' notification area:
![Node-red UI](https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification/assets/31488806/b204c279-0804-43e8-bc96-0d410d5e6050)
To show the sensitivity of the system and the "digital zoom" feature here is the 4K frame for the image in the screen grab of the UI:
![16_36_33 38_TPU_HummingbirdRight_AI](https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification/assets/31488806/5100f77d-2865-43e7-939b-c77c55a29489)

&nbsp;

__Some sample command lines:__
```
TPU initial detection and YOLO8 verification on a "headless" system:
python3 AI.py -d 0 -nsz -nTPU 1 -y8v -rtsp 19cams.rtsp 2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &

TPU initial detection and NCS2 YOLO4-OpenVINO verification on headless system:
source /opt/intel/openvino_2021.3.394/bin/setupvars.sh
python3 AI.py -d 0 -nsz -nTPU 1 -y4ovv -myriad -cam 6onvif.txt -rtsp CW_door.rtsp  2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &

OpenVINO CPU initial detection and GPU YOLO4-OpenVINO verification on Lenovo ThinkPad:
source /opt/intel/openvino_2021.4.752/bin/setupvars.sh
python3 AI.py -d 1 -nsz -nt 1 -y4ovv -rtsp 4UHD.rtsp 2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &
```






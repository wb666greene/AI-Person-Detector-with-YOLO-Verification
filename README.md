# AI-Person-Detector-with-YOLO-Verification
This is the next evolution of https://github.com/wb666greene/AI-Person-Detector
The major change was ading an intermideate output queue to verify detections with a YOLO model and removing the Raspberry Pi support (where YOLO is not viable at present).

Here is an image sequence that shows the software in action, made from the detection images as a solicitor walks from my neighbor's yard, across the street, to my front door, leaves his flyer, and proceeds across my yard on to the next house. He walks across the field of view of multiple cameras with various resolutions of 4K, 5Mpixel, & 1080p:
https://youtu.be/1kDxmbTq4P4

I've had some health issues that laid me up for about a year, but this system has been running 24/7/365 since about September of 2022 with 26 outside cameras each set for 3 fps and has had only four false positives in all that time. Two of the false positives were my neighbor's cat mis-detected as a person that also detected as a person with the YOLO model:
![01_51_24 97_TPU_driveway_AI](https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification/assets/31488806/d67224a1-332f-4e8a-a6cf-611fe2935d15)
![22_02_20 66_TPU_SideYard_AI](https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification/assets/31488806/5f18a4a9-ec20-4220-975d-b801b1cbb042)
and two false positives from out of focus blobs of bugs too close to the camera.  I have a filter in node-red on these cameras to reject detections that are too small or too large to be a real person in the field of view that has stopped most of the bug blob false alerts but not these two:
![01_07_30 57_TPU_CliffwoodDoor_AI](https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification/assets/31488806/25a308d8-0450-4068-9949-0a79cadaf1ae)
![01_16_00 24_TPU_driveway_AI](https://github.com/wb666greene/AI-Person-Detector-with-YOLO-Verification/assets/31488806/34ff6270-5940-483f-bb60-e989d9b98636)

Basic operation is that a thread is launched for each RTSP or ONVIF camera. The MobilenetSSD_v2 AI thread reads the queues round-robin, resizes the frame to 300x300 and if a person is detected a digital zoom is done on the detection which is resized to 300x300 and the inference repeated.  If it exceeds the detection threshold the zoomed image is resized for the YOLO model, currently Darknet YOLO4 ( https://github.com/AlexeyAB/darknet ), Darknet YOLO4 OpenVINO ( https://github.com/TNTWEN/OpenVINO-YOLOV4 ) or Ultralytics YOLO8 ( https://docs.ultralytics.com/ ) and written to an intermediate output queue that does the YOLO inference.  If this detection passes the "person detection" threshold it is written to the final output queue that is read by the main program and passed to the node-red system for "electric fence" and other ad-hoc validity filtering before sending audio and/or MMS and Email alerts depending on the alarm system mode.  All detections are saved independent of alarm system mode and the node-red runs a daily process to remove images older than 30 days.







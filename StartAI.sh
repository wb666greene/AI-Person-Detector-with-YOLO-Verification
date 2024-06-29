#!/bin/bash
# edit for directory of AI code and model directories,
#one symlink beats changing every script and/or multiple node-red nodes
# make symlink to AI directory, for example:
# sudo ln -s /home/wally /home/ai
cd /home/ai/AI

export DISPLAY=:0
export XAUTHORITY=/home/ai/.Xauthority

# should be clean shutdown
usr/bin/pkill -2 -f "AI.py" > /dev/null 2>&1

sleep 5

# but, make sure it goes away before retrying
/usr/bin/pkill -9 -f "AI.py" > /dev/null 2>&1
sleep 1

export PYTHONUNBUFFERED=1
# necessary only if using OpenVINO cv2
##source /opt/intel/openvino/bin/setupvars.sh
##python3 AI.py -nNCS 1 -d 0 -cam onvif.txt  >> ../detect/`/bin/date +%F`_AI.log 2>&1 &
##python3 AI.py -nNCS 1 -nTPU 1 -nt 1 -d 1 -cam onvif.txt  >> ../detect/`/bin/date +%F`_AI.log 2>&1 &

##python3 AI.py -d 1 -z -nTPU 1 -rtsp 4UHD.rtsp  2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &

# make symlink to specify cameras:  ln -s 4UHD.rtsp cameraURL.rtsp or ln -s 6onvif.txt cameraURL.txt
python3 AI.py -d 1 -z -nTPU 1  2>/dev/null >> ../detect/`/bin/date +%F`_AI.log &


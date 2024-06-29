#!/bin/bash
# edit for directory of AI code and model directories,
#one symlink beats changing every script and/or multiple node-red nodes
# make symlink to AIdev directory, for example:
# sudo ln -s /home/wally /home/ai
cd /home/ai/AI

export DISPLAY=:0
export XAUTHORITY=/home/ai/.Xauthority

# should be clean shutdown
/usr/bin/pkill -2 -f "AI.py" > /dev/null 2>&1
sleep 5

sudo /sbin/shutdown -r now

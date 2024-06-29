#!/bin/bash
# edit for directory of AI code and model directories,
#one symlink beats changing every script and/or multiple node-red nodes
# make symlink to AIdev directory, for example:
# sudo ln -s /home/wally /home/ai
cd /home/ai/AI

export DISPLAY=:0
export XAUTHORITY=/home/ai/.Xauthority

# need to change this if using "local save" -ls option
BASEDIR=/home/ai
DETECT=detect

# number of days to save detection images and log files
IMGDAYS=31
LOGDAYS=31
/bin/date
echo "Starting cleanup ..."
/usr/bin/find $BASEDIR/$DETECT/ -maxdepth 1 -type d -mtime +$IMGDAYS -exec rm -rf {} \; >/dev/null 2>&1
/usr/bin/find $BASEDIR/$DETECT/ -maxdepth 1 -type f -mtime +$LOGDAYS -exec rm {} \; >/dev/null 2>&1
/bin/date
echo "Finished"


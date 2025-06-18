#!/bin/bash

while true; do
    # detect if the program is running
    if ! pgrep -f "run.py"; then
        echo "$(date): Program not running. Restarting..." >> program_restart.log
        python run_test.py &
    fi
    sleep 60 # check every minute
done

#!/bin/bash

# Start the Trading Bot in the background
echo "Starting Trading Bot..."
# Redirect stdout and stderr to trade.log (unbuffered)
python -u trade.py &

# Start the Web Interface in the foreground with auto-restart loop
echo "Starting Web Interface..."
while true; do
    python app.py
    echo "Web Interface stopped. Restarting in 1 second..."
    sleep 1
done

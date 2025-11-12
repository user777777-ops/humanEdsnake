
#!/bin/bash

# Start XWayland on display :10 in the background
Xwayland :10 &

# Wait a moment for Xwayland to initialize
sleep 2

# Launch Openbox window manager on display :10
DISPLAY=:10 openbox &

# Launch Mars MIPS IDE on display :10 (replace with correct command or full path)
DISPLAY=:10 kitty &


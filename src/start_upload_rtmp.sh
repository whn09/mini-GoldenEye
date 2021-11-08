ffmpeg -f v4l2 -r 8 -video_size vga -pix_fmt yuv420p12be -i /dev/video0 -b:v 500k -c:v h264_omx -preset ultrafast -an -f flv rtmp://52.81.133.221/live/123456


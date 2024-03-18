# CASE STUDY: AUTOMATED LICENSE PLATE RECOGNITION

___
### HOW TO EXTRACT FRAMES FROM VIDEO STREAM
```shell
ffmpeg -i LicensePlateReaderSample_4K.mov -vcodec mpeg4 -f mpegts -t 60 udp://127.0.0.1:23000
```


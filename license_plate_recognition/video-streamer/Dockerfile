FROM jrottenberg/ffmpeg:4.3-ubuntu
WORKDIR /stream
COPY ./LicensePlateReaderSample_4K.mov /stream/
CMD ["-i", "LicensePlateReaderSample_4K.mov", "-vcodec", "mpeg4", "-f", "mpegts", "udp://127.0.0.1:23000"]

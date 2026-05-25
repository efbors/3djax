ffmpeg -i qam16_constellation.mp4 -vf "scale=800:-1" -vcodec libx264 -crf 28 -an qam16_fixed.mp4

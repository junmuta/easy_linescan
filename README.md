# Easy Linescan

Easily create [linescan-like](https://en.wikipedia.org/wiki/Strip_photography) train photos with a phone camera

## Demo:

Input video:  
![demo video](https://github.com/junmuta/easy_linescan/blob/main/demo/hcmt_epk_smol.gif?raw=true)

Output image:  
![demo output](https://github.com/junmuta/easy_linescan/blob/main/demo/hcmt_epk.png?raw=true)  
Zoomed:
![demo zoomed output](https://github.com/junmuta/easy_linescan/blob/main/demo/hcmt_epk_zoomed.png?raw=true)

Try this out for yourself:  
```python3 trackstrip.py -s demo/hcmt_epk.mp4 -rf```

## How it works:

This program takes vertical slices out of each frame, and putting them next to each other for the output image. 
Unfourtunately, using constant slice widths leads to warping when the speed of the train changes:  
![warped image of xtrapolis train](https://github.com/junmuta/easy_linescan/blob/main/demo/warped_xtrap.png?raw=true)

So motion tracking is used to dynamically change the slice widths to prevent warping.  
Here's how it works:



Firstly, OpenCV's ORB keypoint detector is used to find keypoints in each frame of the video. (line 56, 57)
```        
keypoints = orb.detect(image,None)
keypoints, descriptors = orb.compute(image, keypoints)
```

These keypoints are then searched for in the 4th next frame (depends on configuration), to find matches:  
![matched keypoints on a train](https://github.com/junmuta/easy_linescan/blob/main/diagrams/demo_frame_1-2_matches.png?raw=true)


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

This program takes vertical slices out of each frame, and puts them next to each other for the output image.  
Unfourtunately, using constant slice widths leads to warping when the speed of the train changes:  
![warped image of xtrapolis train](https://github.com/junmuta/easy_linescan/blob/main/demo/warped_xtrap.png?raw=true)

So motion tracking is used to dynamically change the slice widths to prevent warping.

#### Here's how it works:

Firstly, OpenCV's ORB keypoint detector is used to find keypoints in each frame of the video. (line 56, 57)
```        
keypoints = orb.detect(image,None)
keypoints, descriptors = orb.compute(image, keypoints)
```

These keypoints are then searched for in the 4th next frame (depends on configuration), to find matches:  
![matched keypoints from the 2 frames](https://github.com/junmuta/easy_linescan/blob/main/diagrams/demo_frame1-2_matches.png?raw=true)

Plotting the position change of these keypoints between the 2 frames, we get this:  
![Plot of the position delta between the 2 frames](https://github.com/junmuta/easy_linescan/blob/main/diagrams/demo_frame1-2_delta_scatter.png?raw=true)

There are outliers (shown in green and red) in this plot.  
The green ones are removed by repeatedly filtering by the [z-score](https://statisticsbyjim.com/basics/z-score/) of the change in position.  
The red ones are removed by filtering by an absolute limit on movement in the y axis. (this is configurable)  
This assumes that the train only moves horizontally, so hopefully no one is trying to use this to photograph a funicular.

Without the outliers, the plot becomes much clearer, showing 2 distinct groups, circled with red and blue:  
![Plot of the position delta between the 2 frames without the outliers](https://github.com/junmuta/easy_linescan/blob/main/diagrams/demo_frame1-2_delta_scatter_zoomed_annotated.png?raw=true)

The keypoint matches in the blue circle are from keypoints on the station. The movement in the x axis (bottom axis on the plot) can be observed to be around 0.
The keypoint matches in the red circle are the keypoints found on the train. In this case they move around 9 pixels per frame (bottom axis) so this will be the slice width used for this frame when concatenating at the end.

To distinguish the 2 groups, kernel density estimation is used (in the x axis) to create a graph like this:  
![Graph of the density of the change in x](https://github.com/junmuta/easy_linescan/blob/main/diagrams/demo_frame1-2_kde.png?raw=true)

We can clearly see the 2 peaks (one for the station and one for the train), so we can simply find the local maximum turning point with the highest delta x value (bottom axis) to get the slice width for this frame.

The slice width for each frame is found like this, and they're put in a big array that contains the slice widths for each frame.  
They're then processed (missing slice widths filled in, outliers removed, etc).  
This is boring so I won't cover this. Have a look at the sectoin from "def clean_slice_widths" in the code if you are curious.

At this point the slice widths are floating point numbers, but pixels are discrete so they need to be turned into integers.  
Unfourtunately, simply rounding will create warping because:
```[7.3, 7.4, 7.4, 7.3]```
will be turned into
```[7, 7, 7, 7]```
and we lose 1.4 pixels worth of slice widths.

So, the modified rounding algorithm rounds each number, keeps track the difference between the rounded number and the original, and uses that to restore any lost pixels. (line 369)

Now, combining slices from each frame of the original video with the slice widths we found, we get:  
![demo output](https://github.com/junmuta/easy_linescan/blob/main/demo/hcmt_epk.png?raw=true)  
Success!

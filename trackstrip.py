import numpy as np
import os
import cv2
import argparse
import re
import json
from matplotlib import pyplot as plt
import math
from sklearn.neighbors import KernelDensity

parser = argparse.ArgumentParser(
                    prog='Strip2',
                    description='Fake strip photography using high-speed videos')

parser.add_argument('-s', '--specify_video') #video file
parser.add_argument('-r', '--reverse', action="store_true") #reverse each slice
parser.add_argument('-c', '--column')
parser.add_argument('-o', '--output_file')
parser.add_argument('-f', '--flip', action="store_true")
parser.add_argument('-e', '--consecutive_errors_tolerated')
parser.add_argument('-d', '--head_tail_to_discard')
parser.add_argument('-y', '--dy_limit')
parser.add_argument('-m', '--frame_match_dist') # amount of frames to jump for finding matching keypoints in 
parser.add_argument('--kernel_bandwidth') # higher is a smoother kde graph
parser.add_argument('--match_outlier_stdevs')  # stdevs of deviation from mean allowed for matches
parser.add_argument('--width_checking_radius') # amount of frames on either side to consider for mean and stdevs
parser.add_argument('--width_allowed_stdevs') # stdevs of deviation from mean allowed for each frame width
parser.add_argument('--override_multiply_widths')
parser.add_argument('--debug_match', action="store_true")
parser.add_argument('--debug_separation', action="store_true")
parser.add_argument('--output_slice_widths', action="store_true")

args = parser.parse_args()

column = args.column
videoloc = args.specify_video
reverse = args.reverse
output = args.output_file
flip = args.flip
longest_allowed_None = args.consecutive_errors_tolerated
discard_info = args.head_tail_to_discard
debug_match = args.debug_match
debug_separation = args.debug_separation
dy_limit = args.dy_limit
width_multiplier = args.override_multiply_widths
match_frame_dist = args.frame_match_dist
checking_radius = args.width_checking_radius
allowed_stdevs = args.width_allowed_stdevs
output_slice_widths = args.output_slice_widths
match_outlier_stdevs = args.match_outlier_stdevs
kernel_bandwidth = args.kernel_bandwidth

cache_dir = "~/.cache/easy_linescan"

def get_keypoints(videoloc, orb):
    vidcap = cv2.VideoCapture(videoloc)
    success,image = vidcap.read()

    kp_descs = []
    count = 0
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Finding keypoints in frame {count}", end="\r")
        keypoints = orb.detect(image,None)
        keypoints, descriptors = orb.compute(image, keypoints)

        # visualised_kps = cv2.drawKeypoints(image, keypoints, None, color=(255,0,0), flags=0)
        # plt.imshow(visualised_kps)
        # plt.show()
        # input()

        success,image = vidcap.read()
        count += 1
        kp_descs.append((keypoints,descriptors))
        # if count >= 10:
        #     break
    print(f"Finding keypoints in frame {count}")
    return kp_descs

def remove_outliers(show, dxs, dys, matches, stdevs):
    if not stdevs:
        stdevs = 3
    else:
        stdevs = int(stdevs)
    
    dxs_stdev = np.std(dxs)
    dxs_mean = np.mean(dxs)

    outlier_rejected_dxs = []             
    outlier_rejected_dys = []            
    outlier_rejected_matches = []

    no_outlier_dxs = []
    no_outlier_dys = []
    refined_matches = []
    for i, dx in enumerate(dxs):
        if dxs_stdev == 0:
            z = None
        else:
            z = abs((dx-dxs_mean)/dxs_stdev)
        if z == None or z < stdevs: # standard deviations allowed
            no_outlier_dxs.append(dx)
            no_outlier_dys.append(dys[i])
            refined_matches.append(matches[i])
        else:
            if show:
                outlier_rejected_dxs.append(dx)
                outlier_rejected_dys.append(dys[i])
                outlier_rejected_matches.append(matches[i])
    dxs = no_outlier_dxs
    dys = no_outlier_dys
    return dxs, dys, refined_matches, outlier_rejected_dxs, outlier_rejected_dys, outlier_rejected_matches

def find_maximums(e, s): # doesn't do shit about same values next to each other, fix if needed
    maximums = []
    for a in range(len(e)-2):
        i = a+1
        if e[i] >= e[i-1] and e[i] >= e[i+1]:
            maximums.append([e[i], abs(s[i])])
    maximums = sorted(maximums, reverse=True, key=lambda max: max[1])
    return maximums

def match_keypoints(kp_descs, bf, dy_limit, match_frame_dist, match_outlier_stdevs, show=False):

    if not dy_limit:
        dy_limit = 10
    else:
        dy_limit = int(dy_limit)

    if show:
        vidcapf0 = cv2.VideoCapture(videoloc)
        vidcapf1 = cv2.VideoCapture(videoloc)
        # get frame 0
        s0,f0 = vidcapf0.read() 
        # get frame to compare to
        for i in range(match_frame_dist+1):
            s1,f1 = vidcapf1.read() 


    dxses = []
   
    for i in range(len(kp_descs)-match_frame_dist):
        kp_desc0 = kp_descs[i]
        kp_desc1 = kp_descs[i+match_frame_dist]

        if i < len(kp_descs)-(match_frame_dist+1):
            print(f"Finding matches between frames {i} and {i+match_frame_dist}", end="\r")
        else:
            print(f"Finding matches between frames {i} and {i+match_frame_dist}")

        matches = bf.match(kp_desc0[1], kp_desc1[1])

        if show:

            angle_rejected_dxs = []
            angle_rejected_dys = []
            y_rejected_dxs = []
            y_rejected_dys = []

        dxs = []
        dys = []
        for match in matches:
            pt0 = kp_desc0[0][match.queryIdx].pt
            pt1 = kp_desc1[0][match.trainIdx].pt

            dx = (pt1[0]-pt0[0])/match_frame_dist
            dy = (pt1[1]-pt0[1])/match_frame_dist

            # filter by angle
            # limit_angle = 2 # degrees
            # angle = math.atan2(dy, dx)
            # print(angle)

            # if -(abs(abs(angle)-math.pi/2)-math.pi/4) > limit_angle*math.pi/180:
            #     # print("rejected due to angle")
            #     if show:
            #         angle_rejected_dxs.append(dx)
            #         angle_rejected_dys.append(dy)
            #     continue

            # filter by dy
            if abs(dy) > dy_limit:
                if show:
                    y_rejected_dxs.append(dx)
                    y_rejected_dys.append(dy)
                continue

            dxs.append(dx)
            dys.append(dy)

        # filter outliers
        outlier_rejected_dxs = []
        outlier_rejected_dys = []
        outlier_rejected_matches = []
        for i in range(4):
            # dxs, dys, matches, ordxs2, ordys2, orms2= remove_outliers(show, dxs, dys, matches, 5)
            dxs, dys, matches, ordxs2, ordys2, orms2= remove_outliers(show, dxs, dys, matches, match_outlier_stdevs)
            outlier_rejected_dxs += ordxs2
            outlier_rejected_dys += ordys2
            outlier_rejected_matches += orms2

        if show:
            plt.scatter(np.array(dxs), np.array(dys))
            plt.scatter(np.array(angle_rejected_dxs), np.array(angle_rejected_dys))
            plt.scatter(np.array(outlier_rejected_dxs), np.array(outlier_rejected_dys))
            plt.scatter(np.array(y_rejected_dxs), np.array(y_rejected_dys))
            plt.show()

        if show:
            f0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
            f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            img = cv2.drawMatches(f0, kp_desc0[0], f1, kp_desc1[0], matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img)
            plt.show()
            img = cv2.drawMatches(f0, kp_desc0[0], f1, kp_desc1[0], outlier_rejected_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img)
            plt.show()
       
        if show:
            s0,f0 = vidcapf0.read()
            s1,f1 = vidcapf1.read()

        dxses.append(dxs)

    return dxses

def get_slice_widths(dxses, width_multiplier, kernel_bandwidth, show=False):
    if width_multiplier:
        width_multiplier = float(width_multiplier)
    else:
        width_multiplier = 1
    if kernel_bandwidth:
        kernel_bandwidth = float(kernel_bandwidth)
    else:
        kernel_bandwidth = 1

    slice_widths = []
    for dxs in dxses:
        # https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit/35151947#35151947
        if len(dxs) == 0:
            slice_widths.append(None)
            continue
        a = np.array(dxs).reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=kernel_bandwidth).fit(a)
        s = np.linspace(min(dxs),max(dxs))

        e = kde.score_samples(s.reshape(-1,1))
        if show:
            plt.plot(s, e)
            plt.show()

        maximums = find_maximums(e,s)
        if not len(maximums) == 2:
            slice_widths.append(None)
            continue
        unrounded_slice_width = maximums[0][1]*width_multiplier

        # slice_width = round(unrounded_slice_width) # select the first peak, not the second bc the second is the one that's stationary
        slice_width = unrounded_slice_width

        slice_widths.append(slice_width)
    return slice_widths
    
def simplify_slices(slice_widths):
    simplified_slices = []
    for i, width in enumerate(slice_widths):
        if len(simplified_slices) == 0:
            simplified_slices.append([width, 1, [i,i]])
        elif simplified_slices[-1][0] == width:
            simplified_slices[-1][1] += 1
            simplified_slices[-1][2][1] = i
        else:
            simplified_slices.append([width, 1, [i,i]])
    return simplified_slices

def widths_remove_outliers(slice_widths, checking_radius=5, min_certainty=0.5, allowed_stdevs = 1.5):
    if not checking_radius:
        checking_radius = 5
    else:
        checking_radius = int(checking_radius)
    if not allowed_stdevs:
        allowed_stdevs = 1.5
    else:
        allowed_stdevs = float(allowed_stdevs)
    if not min_certainty:
        min_certainty = 0.5
    else:
        min_certainty = float(min_certainty)

    new_slice_widths = slice_widths.copy()

    for i, width in enumerate(slice_widths):
        if width == None:
            new_slice_widths[i] = None
            continue
        start = max(i-5, 0)
        end = min(i+5, len(slice_widths)-1)

        averaging_count = 0
        width_sum = 0
        for j in range(start, end+1):
            if slice_widths[j] != None:
                averaging_count += 1
                width_sum += slice_widths[j]

        # if there are too many Nones
        if averaging_count < min_certainty*(2*checking_radius+1):
            new_slice_widths[i] = None
            continue

        mean = width_sum/averaging_count

        stdev = (sum([(slice_widths[j]-mean)**2 for j in range(start, end+1) if slice_widths[j] != None])/averaging_count)**(0.5)

        if stdev == 0:
            new_slice_widths[i] = None
            continue

        z = abs((width-mean)/stdev)

        if z <= allowed_stdevs:
            new_slice_widths[i] = width
        else:
            new_slice_widths[i] = None
    return new_slice_widths

def clean_slice_widths(slice_widths, longest_allowed_None, discard_info, checking_radius, allowed_stdevs):
    slice_widths = widths_remove_outliers(slice_widths, checking_radius, 0.5, allowed_stdevs)
    simplified_slices = simplify_slices(slice_widths)
    if not longest_allowed_None:
        longest_allowed_None = 10
    else:
        longest_allowed_None = int(longest_allowed_None)

    if not discard_info:
        discard_info = 100 # how many elements to discard before/after end/start
    else:
        discard_info = int(discard_info)

    no_none_slices = []
    for slice_group in simplified_slices:
        if not (slice_group[0] == None and slice_group[1] >= longest_allowed_None):
            no_none_slices.append(slice_group)
      
    cuts = []
    for slice_group in no_none_slices:
        if cuts == []:
            cuts.append([slice_group[2][0], slice_group[2][1]])
        elif slice_group[2][0] == cuts[-1][1]+1:
            cuts[-1][1] = slice_group[2][1]
        else:
            cuts.append([slice_group[2][0], slice_group[2][1]])

    # print(cuts)

    new_slice_widths = slice_widths.copy()

    begin_data = 0
    for i in range(cuts[0][0] + discard_info, len(slice_widths)):
        if not slice_widths[i] == None:
            begin_data = i
            break

    end_data = 0
    for i in range(-(cuts[-1][1]-discard_info), 0):
        j = -i
        if not slice_widths[j] == None:
            end_data = j
            break
        
    for i in range(0, begin_data): # before first cut
        new_slice_widths[i] = slice_widths[begin_data]

    for i in range(end_data+1, len(slice_widths)):
        new_slice_widths[i] = slice_widths[end_data]

    last_non_None = 0 # last non None value
    consecutive_Nones = []
    for i, width in enumerate(new_slice_widths):
        if width == None:
            consecutive_Nones.append(i)
        else:
            if not consecutive_Nones == []:
                delta = width - last_non_None
                dist = len(consecutive_Nones)+1
                for j, pos in enumerate(consecutive_Nones):
                    ratio = (j+1)/dist
                    unratio = (dist-(j+1))/dist
                    perfect_size = last_non_None*unratio+ratio*width
                    new_slice_widths[pos] = perfect_size
            consecutive_Nones = []
            last_non_None = width

    lost = 0
    for i, width in enumerate(new_slice_widths):
        new_width = round(width)
        loss = width-new_width
        lost += loss
        if abs(lost) >= 1:
            gain = math.floor(lost)
            lost = lost - gain
            new_width += gain
        new_slice_widths[i] = new_width

  
    return new_slice_widths

def construct_final_image(slice_widths, videoloc, column, reverse):
    final_width = sum(slice_widths)

    vidcap = cv2.VideoCapture(videoloc)
    success,image = vidcap.read()

    final_height = image.shape[0]
    final_depth = image.shape[2]
    final_image = np.zeros((final_height, final_width, final_depth))
    print(f"Final image shape: {final_image.shape}")

    if not column:
        column = round(image.shape[1]/2)
    else:
        column = int(column)

    count = 0
    pos = 0
    while True:
        print(f"Combining frame {count}, column {pos}", end="\r")
        current_frame_width = slice_widths[count]
        column_frame = image[:, column : column+current_frame_width]
        if reverse:
            column_frame = column_frame[:, ::-1]
        final_image[:, pos : pos+current_frame_width] = column_frame
        count += 1
        if len(slice_widths) == count:
            break
        pos += slice_widths[count-1]
        success,image = vidcap.read()
    print(f"Combining frame {count}, column {pos}")
    return final_image

def verify_cache(cache_path): # cache path should be ~/.cache/easy_linescan/slice_width_data.json
    cache_dir = os.path.dirname(cache_path)
    if not os.path.isdir(cache_dir):
        print(f"The cache directory doesn't exist, creating {cache_dir}")
        os.makedirs(cache_dir)

    if not os.path.isfile(cache_path):
        print(f"The cache file doesn't exist, creating {os.path.basename(cache_path)}")
        with open(cache_path, "w") as f:
            f.write(json.dumps({}))
    return True

def get_sha256sum(file):
    print("Calculating sha256sum hash of video file")
    with open(file, "rb") as f: # b means byte
        bytes = f.read()
        hash = hashlib.sha256(bytes).hexdigest()
    return hash

def write_cache(slice_widths, cache_path, video_hash):
    with open(cache_path, "r") as f:
        cache = json.loads(f.read())
    cache[video_hash] = slice_widths
    with open(cache_path, "w") as f:
        f.write(json.dumps(cache))
    return True

def read_cache(cache_path, video_hash):
    with open(cache_path, "r") as f:
        cache = json.loads(f.read())
    if video_hash in cache:
        print("Loading slices from cache")
        return cache[video_hash]
    else:
        return False

if videoloc:
    orb = cv2.ORB_create() # orb keypoint grabber
    kp_descs = get_keypoints(videoloc, orb)

    bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # brute force keypoint matcher

    if not match_frame_dist:
        match_frame_dist = 4
    else:
        match_frame_dist = int(match_frame_dist)

    dxses = match_keypoints(kp_descs, bf, dy_limit, match_frame_dist, match_outlier_stdevs, show=debug_match)
    slice_widths = get_slice_widths(dxses, width_multiplier, kernel_bandwidth, show=debug_separation)
    if output_slice_widths:
        with open("slice_widths.json", "w") as f:
            f.write(json.dumps(slice_widths))
    # slice_widths = json.loads(open("slice_widths.json").read())
    [None]*math.floor(match_frame_dist/2) + slice_widths + [None]*math.ceil(match_frame_dist/2) 
    slice_widths = clean_slice_widths(slice_widths, longest_allowed_None, discard_info, checking_radius, allowed_stdevs)
    if output_slice_widths:
        with open("processed_slice_widths.json", "w") as f:
            f.write(json.dumps(slice_widths))
    image = construct_final_image(slice_widths, videoloc, column, reverse)
    if not output:
        output = "output.png"

    if flip:
        print("flipping image")
        image = image[:, ::-1]
    
    cv2.imwrite(output, image)
    cv2.waitKey(0)   
    print(f"Image outputted to {output}")

else:
    raise Exception("Video file not specified with -s argument")

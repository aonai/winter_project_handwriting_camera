"""
This file provide helper functions and instructions to train a haar cascade model. 

Useful links:
    https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html
    http://note.sonots.com/SciSoftware/haartraining.html
    https://www.youtube.com/watch?v=XrCAvs9AePM

Instructions: 
1. Create a folder named n. Capture a folder of negative images by pressing 's' when running `object_detection.py` 
    at <root/n>. If number of negative images is not satisfying, use the helper function `download_neg_images()` 
    provided in this file to download images specified in `keywords` from Bing. Run this script to write `neg.txt` 
    that stores path to negative images to be used in training.
2. Create a folder named p. Capture a folder of postive images by pressing 's' when running `object_detection.py` 
    at <root/p>. To obtain a good result, a ratio of 2:1 for postive:negative image counts is suggested. Be sure 
    that distorted or cropped objects are also captured.
3. Create a folder named cascade.    
    The folder structure should now be:
        <root>
            /cascade
            /n
            /p
            codes.py
            neg.txt
3. Run the following command lines one by one at <root> to train a model. Notice that these commands are only available for
    opencv version 3.14 or lower, so it is adviced to make this version of opencv library from source. 

    # Annotate images stored in <root/p>. Follow instructions to draw bounding boxes of objects to detect.
    <path_to_opencv_source_code>/build/bin/opencv_annotation --annotations=pos.txt --images=p/
    
    # Write pos.vec from pos.txt. 
    # `num` should be a number larger than positive image counts.
    # `w` and `h` are width and heights of object in frames.
    <path_to_opencv_source_code>/build/bin/opencv_createsamples -info pos.txt -w 24 -h 24 -num 2000 -vec pos.vec
    
    # Start training. 
    # `numPos` should be number of positive images to be used in training. This number should be around 90% of 
        postive image counts.
    # `numNeg` should be a number larger than negative image counts.
    # `numStages` defined number of stages used during training. Increasing this number will increase accuracy of model
        but will also increase training time.
    # `w` and `h` are width and heights should be the same as the ones used in the previous command.
    <path_to_opencv_source_code>/build/bin/opencv_traincascade -data cascade/ -vec pos.vec -bg neg.txt -numPos 900 -numNeg 600 -numStages 20 -w 24 -h 24
   
~/opencv/opencv-3.4.13/build/bin/opencv_annotation --annotations=pos.txt --images=p/
~/opencv/opencv-3.4.13/build/bin/opencv_createsamples -info pos.txt -w 24 -h 24 -num 2000 -vec pos.vec
~/opencv/opencv-3.4.13/build/bin/opencv_traincascade -data cascade/ -vec pos.vec -bg neg.txt -numPos 900 -numNeg 600 -numStages 20 -w 24 -h 24
"""

import os
import cv2
from icrawler.builtin import BingImageCrawler

def download_neg_images():
    """ Helper class to download negative images using icrawler (https://pypi.org/project/icrawler/).
    Edit `keywords` to change search words in Bing.
    Edit `max_num` in `craw` to change number of images of download. Notice that a link to a image
    may be outdated or invalid and will not be downloaded, so indicating a maximum number larger 
    than desired.
    """
    bing_crawler = BingImageCrawler(storage={'root_dir': 'n'})
    keywords = ['human face', 'human hand', 'room', 'wall']
    filters = dict(size='medium') # comment out this line if wish to download large images as well
    for key in keywords:
        bing_crawler.crawl(keyword=key, max_num=200, file_idx_offset='auto', filters=filters)

if __name__=="__main__": 
    # width and height of frame to be put into haar cascade model. 
    # these should be the save as `self.image_to_save` in `object_detection.py`
    frame_width = 640
    frame_height = 480
    
    # Uncomment the following to download image from Bing
    download_neg_images()

    # resize negative images to preferred frame size and write neg.txt 
    with open('neg.txt', 'w') as f:
        for filename in os.listdir('./n'):
            f.write('n/' + filename + '\n')
            print("neg image = ", filename)
            img = cv2.imread(f'n/{filename}')
            img = cv2.resize(img, (frame_width, frame_height))
            cv2.imwrite(f"n/{filename}", img)




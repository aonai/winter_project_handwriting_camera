"""
This file provides helper functions and instructions to train a haar cascade model. 

- Useful links:
    [Official Opencv document on training](https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html)
    [Tutorial by Naotoshi Seo to train a model to detect faces](http://note.sonots.com/SciSoftware/haartraining.html)
    [Tutorial to train a model used in gaming](https://www.youtube.com/watch?v=XrCAvs9AePM)

- Instructions:     
    1. Create a folder named `n`. Capture a folder of negative images by pressing `s` when running 
        `object_detection.py` at `<root>/n`. If the number of negative images is not satisfying, use the 
        helper function `download_neg_images()` provided in `data_gen.py` to download images specified in 
        `keywords` from Bing. Run `data_gen.py` to write `neg.txt` that stores path to negative images to 
        be used in training.
    2. Create a folder named `p`. Capture a folder of positive images by pressing `s` when running 
        `object_detection.py` at `<root>/p`. To obtain a good result, a ratio of 2:1 for positive:negative 
        image counts is suggested. Be sure that distorted or cropped objects are also captured.
    3. Create a folder named `cascade`. The folder structure should now be:    
        ``` 
        <root>   
            /cascade   
            /n   
            /p   
            codes.py   
            neg.txt   
        ```
    3. Run the following commands one by one at `<root>` to train a model. Notice that these commands are
        only available for OpenCV version 3.14 or lower, so it is advised to make this version of the OpenCV
        library from source. (These command lines prefix are only suitable for Linux system)  
        1. Annotate images stored in `<root>/p` and write annotations to `pos.txt`. Follow instructions to 
            draw bounding boxes of objects to detect.    
            ``` 
            <path_to_opencv_source_code>/build/bin/opencv_annotation --annotations=pos.txt --images=p/
            ```
        
        2. Write `pos.vec` from `pos.txt`.    
            `num` - a number larger than positive image counts.   
            `w` - width of object in frames.   
            `h` - height of object in frames.      
            ```
            <path_to_opencv_source_code>/build/bin/opencv_createsamples -info pos.txt -w 24 -h 24 -num 2000 -vec pos.vec
            ```
        
        3. Start training.    
        `numPos` - the number of positive images to be used in training. This number should be around 90% 
                    of positive image counts.   
        `numNeg` - a number larger than negative image counts.   
        `numStages` - the number of stages used during training. Increasing this number will increase the 
                    accuracy of the model but will also increase training time. 
        `w` - width of object in frames. This should be the same as the one used in the previous command.    
        `h` - height of object in frames. This should be the same as the one used in the previous command.   
            ```
            <path_to_opencv_source_code>/build/bin/opencv_traincascade -data cascade/ -vec pos.vec -bg neg.txt -numPos 900 -numNeg 600 -numStages 20 -w 24 -h 24
            ```

        4. Replace the `cascade/cascade.xml` with the newly trained model to have `object_detection.py` and `tracker.py` track the customized object. 

Examples:
    ~/opencv/opencv-3.4.13/build/bin/opencv_annotation --annotations=pos.txt --images=p/
    ~/opencv/opencv-3.4.13/build/bin/opencv_createsamples -info pos.txt -w 24 -h 24 -num 2000 -vec pos.vec
    ~/opencv/opencv-3.4.13/build/bin/opencv_traincascade -data cascade/ -vec pos.vec -bg neg.txt -numPos 900 -numNeg 600 -numStages 20 -w 24 -h 24
"""

import os
import cv2
from icrawler.builtin import BingImageCrawler

def download_neg_images():
    """ Helper function to download negative images using icrawler (https://pypi.org/project/icrawler/).
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




# Handwriting Classifier with Camera
* Sonia Yuxiao Lai
* 2021 Winter Project

This project is going classify words written in air in front of a depth camera.

## Requirements
* [PyTorch](https://pytorch.org/get-started/locally/)
* [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) 
* [OpenCV](https://opencv.org/#)
* [pyrealsense2](https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html)
* [Intel&reg;RealSense&trade; Depth Camera D435i](https://www.intelrealsense.com/depth-camera-d435i/)

## Contents
1. Machine Learning - [PyTorch Letter Classification](#pytorch-letter-classification)
2. Computer Vision - [Haar Cascade Object Detection](#haar-cascade-object-detection)

## PyTorch Letter Classification

## Haar Cascade Object Detection
This project uses haar cascade train and detect functions provided by opencv to track a pen. The image of the 
tracked portion is shown below.

![pen]()

The haar cascade model is stored in `cascade/cascade.xml`. Run `object_detection.py` to visualize the performance 
of this model. To train a custom haar cascade, `data_gen.py` provides helper functions. The instruction is
also included below.  

#### How to train  a haar cascade model for a custom object:
- Useful links:

    [Official Opencv document on training](https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html)

    [Tutorial by Naotoshi Seo to train a model to detect faces](http://note.sonots.com/SciSoftware/haartraining.html)

    [Tutorial to train a model used in gaming](https://www.youtube.com/watch?v=XrCAvs9AePM)

- Instructions: 

    1. Create a folder named `n`. Capture a folder of negative images by pressing `s` when running `object_detection.py` 
        at `<root/n>`. If number of negative images is not satisfying, use the helper function `download_neg_images()` 
        provided in `data_gen.py` to download images specified in `keywords` from Bing. Run `data_gen.py` to write `neg.txt` that stores path to negative images to be used in training.
    2. Create a folder named `p`. Capture a folder of postive images by pressing `s` when running `object_detection.py` 
        at `<root/p>`. To obtain a good result, a ratio of 2:1 for postive:negative image counts is suggested. Be sure 
        that distorted or cropped objects are also captured.
    3. Create a folder named `cascade`.    
        The folder structure should now be:
            ```
            <root>

                /cascade
            
                /n
            
                /p
            
                codes.py
            
                neg.txt
            ```
    3. Run the following commands one by one at `<root>` to train a model. Notice that these commands are only available for opencv version 3.14 or lower, so it is adviced to make this version of opencv library from source. 

        a. Annotate images stored in `<root/p>`. Follow instructions to draw bounding boxes of objects to detect.
        `<path_to_opencv_source_code>/build/bin/opencv_annotation --annotations=pos.txt --images=p/`
        
        b. Write pos.vec from pos.txt. 
        `num` should be a number larger than positive image counts.
        `w` and `h` are width and heights of object in frames.
        `<path_to_opencv_source_code>/build/bin/opencv_createsamples -info pos.txt -w 24 -h 24 -num 2000 -vec pos.vec`
        
        c. Start training. 
        `numPos` is number of positive images to be used in training. This number should be around 90% of 
        postive image counts.
        `numNeg` should be a number larger than negative image counts.
        `numStages` defines number of stages used during training. Increasing this number will increase accuracy of model
            but will also increase training time.
        `w` and `h` are width and heights. These should be the same as the ones used in the previous command.
        `<path_to_opencv_source_code>/build/bin/opencv_traincascade -data cascade/ -vec pos.vec -bg neg.txt -numPos 900 -numNeg 600 -numStages 20 -w 24 -h 24`
    
    4. Replace the `cascade/cascade.xml` with customized model to have `object_detection.py` and `tracker.py` track the custome object. 
   
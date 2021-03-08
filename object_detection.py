import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import threading, time
import string, random

# ------------- variables ---------------
frame_width = 640
frame_height = 480
kernel = np.ones((5,5),np.uint8)

# ------------ members for streaming -----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, 60)
config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, 60)
profile = pipeline.start(config)

class Detection():
    """ Helper class to use realsense camera vision with haar casacde detections.
    Use this class to visualize the performance of haar cascade model or save 
    frames to be used in training.
    Follow instructions specified in data_gen.py to train a haar cascade model.
    Press 's' to start saving choosen frames (defined in self.image_to_save) for each 0.5 sec.
    """
    def __init__(self):
        # variables for depth camera
        self.background_color = 255
        self.clipping_distance_in_meters = 0.4
        self.min_distance_in_meters = 0

        # variable for saving frames
        self.next_call = time.time()
        self.idx = 0

        # variables for opencv frames
        self.window_name = "Test Detector"
        self.window_images = None

        # initialize
        self.cascade=cv.CascadeClassifier("trainsets_9/cascade/cascade.xml")
        self.setup_stream()
        cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)

        while True:
            self.setup_window()
            cv.imshow(self.window_name, self.window_images)

            # ---------  keys ----------------
            key = cv.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                # Press esc or 'q' to close the image window
                cv.destroyAllWindows()
                break
            elif key & 0xFF == ord('s'):
                self.save_image()

    def setup_stream(self):
        """ Setup streaming for realsense camera """
        # setup streaming 
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        self.clipping_distance = self.clipping_distance_in_meters / depth_scale
        self.min_distance = self.min_distance_in_meters / depth_scale

        # align depth to color frames
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def setup_window(self):
        """ Setup window images to be shown 
        Align depth and colored frames received from intel realsense depth camera, then start tracking a pen
        either using haar cascade object detections.
        Edit `self.window_images` here to change output frame preference.
        Edit `self.image_to_save` to change prefereed frames for training.
        """ 
        # get frames 
        frames = pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame: 
            raise Exception("WARNING: Frame not found")

        # get camera images
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # track pen
        color_image = self.detect(color_image)
        
        self.window_images = np.fliplr(color_image)
        self.image_to_save = color_image
 
    def detect(self, color_image):
        """ Track pen using a haar cascade model
        Detected objects will be bounded by blue boxes. 
        Edit `minSize` and `maxSize` in `detectMultiScale` to define min and max size of object that should be detected.
        Edit `minNeighbors` in `detectMultiScale` to filterout low possibility detections. Increase this number will 
        likely to decrease noises, but true positive images may also be lost.
        Edit `scaleFactor` in `detectMultiScale` to define scaling factor of objects. 
            Args: 
                color_image: frame used for haar cascade detections. This frame should have the same 
                        setup when cascade model is trained. The default is colored RGB image.
        """
        detections = self.cascade.detectMultiScale(color_image, minSize=(20, 20), maxSize=(5000,50000), minNeighbors=8)
        for (x,y,w,h) in detections:
            color_image = cv.rectangle(color_image,(x,y),(x+w,y+h),(255,0,0), 2)
        return color_image

    def save_image(self):
        """ Helper function to save image for each 0.5 seconds 
        https://stackoverflow.com/questions/8600161/executing-periodic-actions-in-python
        The frame that is saved is defined by `self.save_image`. Edit this variable at end of `self.setup_window`.
        """
        print("saving image #", self.idx)
        self.idx += 1
        letters = string.ascii_lowercase
        rnd_str = ( ''.join(random.choice(letters) for i in range(10)) )
        cv.imwrite(f'image_{self.idx}_{rnd_str}.png',self.image_to_save)
        
        # change 0.5 to other seconds if wish to call this function at a different time loop
        self.next_call = self.next_call+0.5     
        threading.Timer(self.next_call - time.time(), self.save_image).start()


def main():
    """ The main() function. """
    detection = Detection()

if __name__=="__main__": 
    try:
        main()
    except Exception as inst:
            print(inst)
    finally:
        pipeline.stop()
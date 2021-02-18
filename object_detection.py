import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import datetime, threading, time




# ------------- variables ---------------
frame_width = 640
frame_height = 480
kernel = np.ones((5,5),np.uint8)

# ------------ members for streaming -----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, 30)
profile = pipeline.start(config)

cascade=cv.CascadeClassifier("cascade_50x50/cascade.xml")
back_sub = cv.createBackgroundSubtractorKNN()

class Detection():
    """ helper class to handle realsen camera vision """
    def __init__(self):
        self.next_call = time.time()
        self.idx = 0
        self.background_color = 255
        self.clipping_distance_in_meters = 0.4
        self.min_distance_in_meters = 0

        self.window_name = "Detector"
        self.window_images = None

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


        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= self.min_distance), \
                            self.background_color, color_image)
             
        mask, color_mask = self.bg_sub(bg_removed, color_image)
        # color_image = self.detect(mask, color_image)
        
        # depth and color images combined 
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET) 
        
        # stack images
        # self.window_images = np.hstack((bg_removed, color_image, res))
        self.window_images = np.hstack((mask, color_mask, np.fliplr(color_image)))
        self.image_to_save = color_mask
 
    def detect(self, mask, color_image):
        detections = cascade.detectMultiScale(mask, scaleFactor=2, minNeighbors=3, 
                                                minSize=(50,50), maxSize=(100,100))
        for (x,y,w,h) in detections:
            color_image = cv.rectangle(color_image,(x,y),(x+w,y+h),(255,0,0),2)
        return color_image

    def bg_sub(self, bg_removed, color_image):
        mask = back_sub.apply(color_image)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # grey
        mask_bw = cv.cvtColor(mask,cv.COLOR_GRAY2RGB)

        # rgb 
        mask_inv = cv.bitwise_not(mask)
        gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
        rows, cols, channels = color_image.shape
        image = color_image[0:rows, 0:cols]
        colored_portion = cv.bitwise_or(image, image, mask = mask)
        colored_portion = colored_portion[0:rows, 0:cols]
        gray_portion = cv.bitwise_or(gray, gray, mask = mask_inv)
        gray_portion = np.stack((gray_portion,)*3, axis=-1)
        color_mask = colored_portion + gray_portion
        color_mask = cv.GaussianBlur(color_mask,(5,5),0)

        # increase brightness
        brightness = 50
        hsv = cv.cvtColor(color_mask, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        lim = 255 - brightness
        v[v > lim] = 255
        v[v <= lim] += brightness
        final_hsv = cv.merge((h, s, v))
        color_mask = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
 
        return mask_bw, color_mask

    def save_image(self):
        print("saving image #", self.idx)
        self.next_call = self.next_call+1
        threading.Timer( self.next_call - time.time(), self.save_image).start()
        cv.imwrite(f'trainsets_4/test_image_{self.idx}.png',self.image_to_save)
        self.idx += 1
    


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

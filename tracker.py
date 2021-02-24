import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import time


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


class Tracker():
    """ helper class to handle realsen camera vision 
    This class should setup intel realsense camera streaming, allow setting up filter color
    range using trackbars, track a pen by either color filtering or haar cascade detections to
    writing on the frames, and finally output preditected letters written on the frames. 
    """
    def __init__(self):
        # variables for depth camera
        self.background_color = 255
        self.clipping_distance_in_meters = 0.4
        self.min_distance_in_meters = 0
        self.pen_min_distance = 2.5

        # variable for pen tracking
        self.pen = [None, None]
        self.pen_x = []
        self.pen_y = []
        self.board = None
        self.board_rect = None
        self.rect_min = (int(frame_height/5))
        self.rect_max = (int(frame_height*4/5))
        self.write_time = time.time()        
        self.cascade=cv.CascadeClassifier("cascade/cascade.xml")

        # variables for opencv frames
        self.window_name = "Track Pen"
        self.window_images = None

        # initialize
        self.setup_stream()
        cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)

        # Uncomment the following to define range of color to filter using trackbar
        # self.enable_trackbar() 
        
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
                self.save_board()
            elif key & 0xFF == ord('c'):
                print("clear board")
                self.clear_board()
                
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

    def enable_trackbar(self):
        """ Setup trackbar at top of window 
        Call this function when initializing the class will allow user to define range of filter colors manually
        """
        cv.createTrackbar('Lower H', self.window_name, 110, 180, lambda _: None) 
        cv.createTrackbar('Lower S', self.window_name, 120, 255, lambda _: None) 
        cv.createTrackbar('Lower V', self.window_name, 120, 255, lambda _: None)
        cv.createTrackbar('Upper H', self.window_name, 130, 180, lambda _: None) 
        cv.createTrackbar('Upper S', self.window_name, 255, 255, lambda _: None) 
        cv.createTrackbar('Upper V', self.window_name, 255, 255, lambda _: None)

    def get_tracekbar_val(self):
        """ Helper class to filter color when using trackbars 
        
            Returns:
                lower_color: lower bound of filter color range in np.array([h, s, v]) format
                upper_color: upper bound of filter color range in np.array([h, s, v]) format
        """
        lH = cv.getTrackbarPos('Lower H', self.window_name)
        lS = cv.getTrackbarPos('Lower S', self.window_name)
        lV = cv.getTrackbarPos('Lower V', self.window_name)
        uH = cv.getTrackbarPos('Upper H', self.window_name)
        uS = cv.getTrackbarPos('Upper S', self.window_name)
        uV = cv.getTrackbarPos('Upper V', self.window_name)
        
        lower_color = np.array([lH,lS,lV])
        upper_color = np.array([uH,uS,uV])

        # Edit the following numbers to define your range of color to filter
        # The default color range is red 
        lower_color = np.array([135,110,50])
        upper_color = np.array([180,250,150])

        return lower_color, upper_color
        
    def setup_window(self):
        """ Setup window image to be shown 
        align depth and colored frames received from intel realsense depth camera, then start tracking a pen
        either using color filtering or haar cascade object detections to write on frames. 

        Edit self.window_images here to change output frame preference.
        Calling self.write or self.write_color_filter can start writing with a pen in front of the camera.
        """ 
        # get frames 
        frames = pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        self.depth_frame = aligned_depth_frame
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame: 
            raise Exception("WARNING: Frame not found")

        # get camera images
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # setup bounding box for writing
        if self.board is None:
            self.board = np.zeros_like(color_image)
            self.board_rect = np.zeros_like(color_image)
            self.board_rect = cv.rectangle(self.board_rect,(self.rect_min,self.rect_min),(self.rect_max,self.rect_max),(0,255,25), 10)
        
        # Uncomment the following to use frames with depth
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= self.min_distance), \
        #                     self.background_color, color_image)
             
        # Uncomment the following to track a pen by filtering coler, should be used with depth frames
        # hsv = cv.cvtColor(bg_removed, cv.COLOR_BGR2HSV)
        # mask = self.filter_color(hsv)
        # color_image = cv.add(color_image, self.board_rect)
        # color_image = self.write_color_filter(mask, color_image)
        # mask_bw = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) # back and white images 
        # res = cv.bitwise_and(bg_removed, bg_removed, mask=mask)

        # Uncomment the following to track a pen using haar cascade object detection
        color_image = cv.add(color_image, self.board_rect)
        color_image = self.write(color_image)
        
        # Edit self.window_images to change your preferred output frame
        # self.window_images = np.hstack((bg_removed, np.fliplr(color_image), res)) # frames for color filtering
        self.window_images = np.fliplr(color_image)                                 # frames for haar cascade detections
    
    def filter_color(self, hsv):
        """ Setup mask on a frame to extract a specified color range from self.get_trackbar_val()
        
            Args:
                hsv: frame in hsv format
            Returns:
                mask: mask with fitlered color
        """
        lower_color, upper_color = self.get_tracekbar_val()
        mask = cv.inRange(hsv, lower_color, upper_color)
        # filter white nosie
        # https://stackoverflow.com/questions/42272384/opencv-removal-of-noise-in-image
        mask = cv.dilate(mask, kernel, iterations = 1)
        mask = cv.erode(mask, kernel, iterations = 1)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        return mask

    def write(self, image):
        """ Write on a board using pen tracked by haar cascade detections. 
        The default haar cascade model is stored in `cascade/cascade.xml`

            Usage:
                Move away from the camera to indicate lifting the pen.
                Press 's' to save the current board to `test_image.pgn` .
                Pres 'c' to completely clear the writing board.

            Args: 
                image: frame used for haar cascade detections. This frame should have the same 
                        setup when cascade model is trained. The default is colored RGB image.
        """
        detections = self.cascade.detectMultiScale(image, minSize=(20, 20), maxSize=(80,80), minNeighbors=8)
        for (x,y,w,h) in detections:
            image = cv.circle(image, (x, y), 15, (255,0,0), -1)

            if x < self.rect_max and x > self.rect_min and y < self.rect_max and y > self.rect_min \
                and self.depth_frame.get_distance(int(x+w/2), int(y+h/2)) < self.pen_min_distance:
                if self.pen[0] is None:
                    self.pen[0] = (x, y)
                    self.pen[1] = (x, y)
                else:
                    self.pen[0] = self.pen[1]
                    self.pen[1] = (x, y)
        
                if time.time() - self.write_time  > 1:
                    self.pen = [None, None]
                self.write_time = time.time()

            break
        # write 
        if not self.pen[0] is None:
            self.board = cv.line(self.board, self.pen[0], self.pen[1], (255,255, 255), 20)
            image = cv.add(image, self.board)
        return image
    
    def write_color_filter(self, mask, image):
        """ Write on a board using pen tracked by color filtering.
        Notice that this method does not have functionality to lift the pen. A pen is only recognized 
        if the colored contour area is larger than 800 pixels.Due to large noises in color filtering, 
        pen locations will be averaged on each 10 frames. 

            Usage:
                Press 's' to save the current board to `test_image.pgn`.
                Pres 'c' to completely clear the writing board.

            Args: 
                mask: frame used for colored pen tracking. This frame should be already filtered 
                        using desired color range.
                image: frame to show writing on.
        """
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if contours and cv.contourArea(max(contours, key=cv.contourArea)) > 800:
            c = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(image, (x, y), (x+w, y+h), (0,25, 255), 2)

            if x < self.rect_max and x > self.rect_min and y < self.rect_max and y > self.rect_min:
                self.pen_x.append(x)
                self.pen_y.append(y)
            else:
                self.pen_x = []
                self.pen_y = []

            # average pen location on each 10 frames
            if len(self.pen_x) == 10:
                self.pen[0] = self.pen[1]
                self.pen[1] = (int(np.average(self.pen_x)), int(np.average(self.pen_y)))
                self.pen_x = []
                self.pen_y = []
        # write
        if not self.pen[0] is None:
            self.board = cv.line(self.board, self.pen[0], self.pen[1], (255,255, 255), 20)
            image = cv.add(image, self.board)

        return image

    def save_board(self):
        """ Helper function to save writing board """
        print("saving writing")
        cv.imwrite('test_image.png',self.board[self.rect_min:self.rect_max, self.rect_min:self.rect_max])
    
    def clear_board(self):
        """ Helper function to completely clear wiriting board """
        self.board = None
        self.pen = [None, None]


def main():
    """ The main() function. """
    tracker = Tracker()

if __name__=="__main__": 
    try:
        main()
    except Exception as inst:
            print(inst)
    finally:
        pipeline.stop()

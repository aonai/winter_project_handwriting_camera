import pyrealsense2 as rs
import numpy as np
import cv2 as cv

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
    """ helper class to handle realsen camera vision """
    def __init__(self):
        self.background_color = 255
        self.clipping_distance_in_meters = 0.3
        self.min_distance_in_meters = -0.3

        self.pen = [None, None]
        self.board = None

        self.window_name = "Track Pen"
        self.window_images = None

        self.setup_stream()
        cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)
        # find range of color to filter 
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
                print("saving writing")
                self.save_board()
                
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

    def enable_trackbar(self):
        cv.createTrackbar('Lower H', self.window_name, 110, 180, lambda _: None) 
        cv.createTrackbar('Lower S', self.window_name, 120, 255, lambda _: None) 
        cv.createTrackbar('Lower V', self.window_name, 120, 255, lambda _: None)
        cv.createTrackbar('Upper H', self.window_name, 130, 180, lambda _: None) 
        cv.createTrackbar('Upper S', self.window_name, 255, 255, lambda _: None) 
        cv.createTrackbar('Upper V', self.window_name, 255, 255, lambda _: None)

    def get_tracekbar_val(self):
        lH = cv.getTrackbarPos('Lower H', self.window_name)
        lS = cv.getTrackbarPos('Lower S', self.window_name)
        lV = cv.getTrackbarPos('Lower V', self.window_name)
        uH = cv.getTrackbarPos('Upper H', self.window_name)
        uS = cv.getTrackbarPos('Upper S', self.window_name)
        uV = cv.getTrackbarPos('Upper V', self.window_name)
        
        lower_color = np.array([lH,lS,lV])
        upper_color = np.array([uH,uS,uV])

        # red range
        lower_color = np.array([135,110,50])
        upper_color = np.array([180,250,150])

        return lower_color, upper_color
        
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
        
        if self.board is None:
            self.board = np.zeros_like(color_image)

        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= self.min_distance), \
                            self.background_color, color_image)
             
        # filter red coler
        hsv = cv.cvtColor(bg_removed, cv.COLOR_BGR2HSV)
        mask = self.filter_color(hsv)

        # draw
        color_image = self.draw(mask, color_image)

        mask_bw = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) # back and white images 
        res = cv.bitwise_and(bg_removed, bg_removed, mask=mask)

        # depth and color images combined 
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET) 
        
        # stack images
        self.window_images = np.hstack((bg_removed, color_image, res))
    
    def filter_color(self, hsv):
        lower_color, upper_color = self.get_tracekbar_val()
        mask = cv.inRange(hsv, lower_color, upper_color)
        # filter white nosie
        # https://stackoverflow.com/questions/42272384/opencv-removal-of-noise-in-image
        mask = cv.dilate(mask, kernel, iterations = 1)
        mask = cv.erode(mask, kernel, iterations = 1)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        return mask

    def draw(self, mask, image):
        """
        Draw rectangle around pen and save location of pen tip
            mask - image mask that is filter by a choosen color
            image - image to draw contours on
        """
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if contours and cv.contourArea(max(contours, key=cv.contourArea)) > 500:
            c = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(image, (x, y), (x+w, y+h), (0,25, 255), 2)

            # TODO: filter noises here 
            self.pen[0] = self.pen[1]
            self.pen[1] = (x, y)
        
        if not self.pen[0] is None:
            self.board = cv.line(self.board, self.pen[0], self.pen[1], (0,25, 255), 5)
            image = cv.add(image, self.board)

        return image

    def save_board(self):
        cv.imwrite('test_image.png',self.board)
        


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
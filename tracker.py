import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import time
from net_classifier import Classifier
import os, random, string 
from shutil import copyfile
import sys, getopt
import fcntl
from v4l2 import (
    v4l2_format, VIDIOC_G_FMT, V4L2_BUF_TYPE_VIDEO_OUTPUT, V4L2_PIX_FMT_RGB24,
    V4L2_FIELD_NONE, VIDIOC_S_FMT
)

# ------------- variables ---------------
frame_width = 640
frame_height = 480
frame_rate = 60


class Tracker():
    """ Class to handle camera vision 
    This class should be able to setup intel realsense camera streaming, allow setting up filter color
    range using trackbars, track a pen by either color filtering or haar cascade detections to
    writing on the frames, and finally output predicted letters written on the frames. 

        Args:
            use_web_cam: whether to use webcam for tracking pen.
            use_colored_pen: whether to track pen using color filtering 
            use_default_model: whether to use the default trained model `models/model_letters.pth`.
                            if this is set to False, specify a location of the customized model
                            using `model_path`.
            model_path: path to PyTorch model.
            enable_stream: whether to enable streaming to a virtual camera.
            virtual_cam: address of the virtual camera to stream to. 
                            
    """
    def __init__(self, use_web_cam = False, use_colored_pen=False, use_default_model = True, model_path = None,  
                    enable_stream = False, virtual_cam = "/dev/video11"):
       
        # use webcam or realsense depth camera
        self.use_web_cam =  use_web_cam
        if use_web_cam:
            self.pipeline = cv.VideoCapture(0) 
        else:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, frame_rate)
            config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, frame_rate)
            self.profile = self.pipeline.start(config)
        
        # variables for depth camera
        self.background_color = 255
        self.clipping_distance_in_meters = 0.4
        self.min_distance_in_meters = 0
        self.pen_min_distance = 2.5
        self.fps = 0

        # variables for pen tracking
        self.use_colored_pen = use_colored_pen
        if self.use_colored_pen:
            self.backSub = cv.createBackgroundSubtractorKNN()
        self.pen = [None, None]
        self.pen_bound = [[frame_width, frame_height], [0,0]]
        self.pen_x = []
        self.pen_y = []
        self.board = None
        self.board_rect = None
        self.rect_min = (int(frame_height/5))
        self.rect_max = (int(frame_height*4/5))
        self.write_time = time.time()        
        self.cascade=cv.CascadeClassifier("cascade/cascade.xml")
        self.last_call = time.time()
        self.pen_color = (255,255,255)
        self.pen_size = 20

        # pytorch classifier
        self.classifier = Classifier() if use_default_model else Classifier(model_path)
        self.predictions = []
        self.pred_idx = 0
        self.image_path ='user_images'
        self.image_dataset = 'user_dataset'

        # variables for opencv frames
        self.window_name = "Track Pen"
        self.window_images = None

        # initialize
        if (not self.use_web_cam) : self.setup_stream()
        cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)
        self.insturctions = None

        # setup streaming
        if enable_stream: 
            self.virtual_cam = virtual_cam
            self.setup_virtual_cam()

        self.start_time = time.time()
        while True:
            self.setup_window()
            # cv.imshow("Instructions", self.insturctions)      # Uncomment this line to show instructions
            cv.imshow(self.window_name, self.window_images)
            
            if enable_stream: 
                # Convert frame to RGB format
                virtual_image = cv.UMat.get(self.window_images)
                virtual_image = cv.cvtColor(virtual_image, cv.COLOR_BGR2RGB)
                # write to output virtual cam loopback
                written = os.write(self.virtual_cam_output, virtual_image.data)
                if written < 0:
                    print("ERROR: could not write to output device!")
                    os.close(self.virtual_cam_output)
                    break

            # ---------  keys ----------------
            key = cv.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                # Press `esc` or `q` to close the image window
                cv.destroyAllWindows()
                if (use_web_cam):
                    self.pipeline.release()
                else:
                    self.pipeline.stop()
                break
            elif key & 0xFF == ord('s'):
                # Press `s` to save writing and predict letter
                print("Classify")
                self.save_board()
            elif key & 0xFF == ord('a'):
                # Press `a` to save current image and prediction to user dataset
                print("Save to model")
                self.save_model()
            elif key & 0xFF == ord('w'):
                # Press `w` to clear predictions displayed on the screen
                print("Clear predictions")
                self.predictions = []
            elif key & 0xFF == ord('c'):
                # Press `c` to clear writing board
                print("Clear board")
                self.clear_board()
            elif key & 0xFF == ord('n'):
                # Press `n` to redo false prediction
                print("Redo prediction")
                self.redo_classify()
            elif key & 0xFF == ord('e'):
                # Press `e` to switch between pen and eraser
                if self.pen_size == 20:
                    print("Eraser")
                    self.pen = [None, None]
                    self.pen_color = (0, 0, 0)
                    self.pen_size = 30
                else:
                    print("Pen")
                    self.pen = [None, None]
                    self.pen_color = (255,255, 255)
                    self.pen_size = 20

    def setup_virtual_cam(self):
        """ Setup v4l2 lookback config 
        The following code is adapted from 
        https://arcoresearchgroup.wordpress.com/2020/06/02/virtual-camera-for-opencv-using-v4l2loopback/ 
        """
        print("Setup virtual camera at ", self.virtual_cam)
        # open output device
        try:
            self.virtual_cam_output = os.open(self.virtual_cam, os.O_RDWR)
        except Exception as ex:
            print("ERROR: could not open output device!")
            print(str(ex))
            return -1

        # configure params for output device
        vid_format = v4l2_format()
        vid_format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT
        if fcntl.ioctl(self.virtual_cam_output, VIDIOC_G_FMT, vid_format) < 0:
            print("ERROR: unable to get video format!")
            return -1

        framesize = frame_width * frame_height * 3
        vid_format.fmt.pix.width = frame_width
        vid_format.fmt.pix.height = frame_height

        # output frames need to be in rgb format
        vid_format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24
        vid_format.fmt.pix.sizeimage = framesize
        vid_format.fmt.pix.field = V4L2_FIELD_NONE

        if fcntl.ioctl(self.virtual_cam_output, VIDIOC_S_FMT, vid_format) < 0:
            print("ERROR: unable to set video format!")
            return -1

                
    def setup_stream(self):
        """ Setup streaming for realsense camera """
        # setup streaming 
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        self.clipping_distance = self.clipping_distance_in_meters / depth_scale
        self.min_distance = self.min_distance_in_meters / depth_scale

        # align depth to color frames
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def enable_trackbar(self):
        """ Setup trackbar at top of window 
        Call this function when initializing the class will allow user to define range of filter colors manually.
        """
        cv.createTrackbar('Lower H', self.window_name, 110, 180, lambda _: None) 
        cv.createTrackbar('Lower S', self.window_name, 120, 255, lambda _: None) 
        cv.createTrackbar('Lower V', self.window_name, 120, 255, lambda _: None)
        cv.createTrackbar('Upper H', self.window_name, 130, 180, lambda _: None) 
        cv.createTrackbar('Upper S', self.window_name, 255, 255, lambda _: None) 
        cv.createTrackbar('Upper V', self.window_name, 255, 255, lambda _: None)

    def get_tracekbar_val(self):
        """ Filter color when using trackbars 
        
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
        Align depth and colored frames received from intel realsense depth camera, then start tracking a pen
        either using color filtering or haar cascade object detection to write on frames. 

        Edit self.window_images here to change output frame preference.
        """ 
        # get frames 
        if (not self.use_web_cam):
            frames = self.pipeline.wait_for_frames()
            num_frames = frames.get_frame_number()
            self.fps = num_frames/(time.time() - self.start_time)
            
            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame() 
            self.depth_frame = aligned_depth_frame
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame: 
                raise Exception("WARNING: Frame not found")

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
        else:
            ret, color_image = self.pipeline.read()

        # setup board for writing
        if self.board is None:
            self.board = np.zeros_like(color_image)
        self.board_rect = np.zeros_like(color_image)
        self.board_rect = cv.rectangle(self.board_rect,(self.pen_bound[0][0]-50,self.pen_bound[0][1]-50), \
                                        (self.pen_bound[1][0]+50,self.pen_bound[1][1]+50),(0,255,25), 10)
        
        # Uncomment the following to display menu on the screen
        # self.setup_menu(color_image)
        # cv.imwrite("instructions.png", self.insturctions)

        # Uncomment the following to use frames with depth
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= self.min_distance), \
        #                     self.background_color, color_image)
             
        if self.use_colored_pen:    #track a pen by detecting a coler range 
            color_image = self.back_sub(color_image)
            hsv = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)
            mask = self.filter_color(hsv)
            color_image = self.write_color_filter(mask, color_image)
            mask_bw = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) # back and white images 
            res = cv.bitwise_and(color_image, color_image, mask=mask)
        else:       #track a pen using haar cascade object detection
            color_image = self.write(color_image)

        # Put classified letters on frames
        color_image = np.fliplr(color_image)
        color_image = cv.UMat(color_image)         
        pred_str = ' '.join(self.predictions)
        font = cv.FONT_HERSHEY_SIMPLEX
        loc = (100, 150)
        fontScale = 1.5
        thickness = 3
        color = (255, 0, 0)
        color_image = cv.putText(color_image, pred_str, loc, font, fontScale, color, thickness, cv.LINE_AA) 

        # Uncomment the following line to print fps on window
        # color_image = cv.putText(color_image, f"fps = {str(self.fps)}", (frame_width-200, frame_height-30), 
        #                             font, 1, (0, 0, 255), 1, cv.LINE_AA) 
        

        # Edit self.window_images to change your preferred output frame
        self.window_images = color_image                                        
    
    def back_sub(self, frame):
        """ Background subtraction
        The following code is adapted from 
        https://automaticaddison.com/how-to-apply-a-mask-to-an-image-using-opencv/

            Returns:
                output: frames overlayed with a grey mask on the steady background.
        """
        mask = self.backSub.apply(frame)
        mask_inv = cv.bitwise_not(mask)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rows, cols, channels = frame.shape
        frame = frame[0:rows, 0:cols]
        colored_portion = cv.bitwise_or(frame, frame, mask = mask)
        colored_portion = colored_portion[0:rows, 0:cols]
        gray_portion = cv.bitwise_or(gray, gray, mask = mask_inv)
        gray_portion = np.stack((gray_portion,)*3, axis=-1)
        output = colored_portion + gray_portion
        return output

    def filter_color(self, hsv):
        """ Setup mask on a frame to extract a specified color range from self.get_trackbar_val()
        
            Args:
                hsv: frame in hsv format.
            Returns:
                mask: mask with fitlered color.
        """
        lower_color, upper_color = self.get_tracekbar_val()
        mask = cv.inRange(hsv, lower_color, upper_color)
        # filter white nosie, adapted from https://stackoverflow.com/questions/42272384/opencv-removal-of-noise-in-image
        kernel = np.ones((5,5),np.uint8)
        mask = cv.dilate(mask, kernel, iterations = 1)
        mask = cv.erode(mask, kernel, iterations = 1)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        return mask

    def write(self, image):
        """ Write on a board using a pen tracked by haar cascade object detection. 
        The default haar cascade model is stored in `cascade/cascade.xml`

            Args: 
                image: frame used for haar cascade detections. This frame should have the same 
                        setup when the cascade model is trained. The default is colored RGB frame.
            Returns:
                image: image with writtings displayed.
        """
        # detect
        detections = self.cascade.detectMultiScale(image, minSize=(30, 30), maxSize=(80, 80), minNeighbors=10)
        for (x,y,w,h) in detections:
            if self.pen[0] is None or (abs(self.pen[1][0]-x)<=2*w and abs(self.pen[1][1]-y)<=2*h):
                image = cv.circle(image, (x, y), 15, (255, 0, 0), -1)
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
            self.board = cv.line(self.board, self.pen[0], self.pen[1], self.pen_color, self.pen_size)
            self.pen_bound[0][0] = min(self.pen_bound[0][0], self.pen[1][0])
            self.pen_bound[0][1] = min(self.pen_bound[0][1], self.pen[1][1])
            self.pen_bound[1][0] = max(self.pen_bound[1][0], self.pen[1][0])
            self.pen_bound[1][1] = max(self.pen_bound[1][1], self.pen[1][1])
            image = cv.add(image, self.board)
        return image
    
    def write_color_filter(self, mask, image):
        """ Write on a board using a pen tracked by color filtering.
        Notice that this method does not have the functionality to lift the pen. A pen is only recognized 
        if the colored contour area is larger than 500 pixels.Due to large noises in color filtering, 
        pen locations will be averaged on every 5 frames. 

            Args: 
                mask: frame used for colored pen tracking. This frame should be already filtered 
                        using desired color range.
                image: frame to show writing on.
            Returns:
                image: image with writtings displayed.
        """
        # detect
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if contours and cv.contourArea(max(contours, key=cv.contourArea)) > 500:
            c = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(image, (x, y), (x+w, y+h), (0,25, 255), 2)

            self.pen_x.append(x)
            self.pen_y.append(y)

            # average pen location on every 5 frames
            if len(self.pen_x) == 5:
                self.pen[0] = self.pen[1]
                self.pen[1] = (int(np.average(self.pen_x)), int(np.average(self.pen_y)))
                self.pen_x = []
                self.pen_y = []
            # write
            if not self.pen[0] is None:
                self.board = cv.line(self.board, self.pen[0], self.pen[1], (255,255, 255), 20)
                self.pen_bound[0][0] = min(self.pen_bound[0][0], self.pen[1][0])
                self.pen_bound[0][1] = min(self.pen_bound[0][1], self.pen[1][1])
                self.pen_bound[1][0] = max(self.pen_bound[1][0], self.pen[1][0])
                self.pen_bound[1][1] = max(self.pen_bound[1][1], self.pen[1][1])
    
        else:
            self.pen = [None, None]
        image = cv.add(image, self.board)
        return image

    def save_board(self):
        """ Save writing board 
        The default user image location is `user_images/`. Calling this function will 
        save the current board as the last file in folder. 
        """
        # save writing board to user_images
        path = self.image_path
        files = os.listdir(path)
        num = len(files)
        letters = string.ascii_lowercase
        rnd_str = ( ''.join(random.choice(letters) for i in range(10)) )
        file_name = f'{num}_{rnd_str}.png'
        thresh = 30
        from_x = self.pen_bound[0][0] - thresh
        from_y = self.pen_bound[0][1] - thresh
        to_x = self.pen_bound[1][0] + thresh
        to_y = self.pen_bound[1][1] + thresh
        board_to_save = self.board[from_y:to_y, from_x:to_x]
        cv.imwrite(f"{path}/{file_name}", board_to_save)
        print("Save writing ", file_name)

        # classify letter on board
        predicted = self.classifier.classify(file_name)
        print("Prediction = ", predicted)
        self.predictions.append(predicted)
        self.pred_idx = 0

        self.clear_board()
    
    def redo_classify(self):
        """ Redo the last letter prediction """
        self.predictions.pop()
        path = self.image_path
        files = os.listdir(path)
        num = len(files)
        self.pred_idx += 1
        predicted = self.classifier.classify(str(num-1), self.pred_idx)
        self.predictions.append(predicted)
    
    def clear_board(self):
        """ Completely clear wiriting board """
        self.board = None
        self.pen = [None, None]
        self.pen_bound = [[frame_width, frame_height], [0,0]]

    def setup_menu(self, image):
        """ Setup menu as an image that has the same size of image """
        if self.insturctions is None:
            self.insturctions = np.zeros_like(image, np.uint8)
            self.insturctions = cv.UMat(self.insturctions)
            font = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            thickness = 1
            color = (255, 255, 255)
            text = ["Press 'Esc' or 'q' to exit", 
                    "Press 'c' to completely clear board",
                    "Press 'w' to clear predictions",
                    "Move away from camera to lift pen",
                    "Press 's' to save board and classify letter",
                    "Press 'a' to save current prediction to dataset",
                    "Press 'n' to indicate wrong prediction and redo",
                    "Press 'e' to switch between eraser and pen"]
            self.insturctions = cv.putText(self.insturctions, "Insutrctions:", (20, 50), font, 1, color, 2, cv.LINE_AA) 
            for i, t in enumerate(text):
                self.insturctions = cv.putText(self.insturctions, t, (20, 90+30*i), font, fontScale, color, thickness, cv.LINE_AA)  

    def save_model(self):
        """ Save current prediction to user dataset 
        The default path to store the image is user_dataset/
        """ 
        path = self.image_path
        files = os.listdir(path)
        num = len(files)
        for f in files:
            if f.startswith(str(num-1)):
                print("old file name ", f)
                dataset = os.listdir(self.image_dataset)
                dataset_num = len(dataset)
                file_name = f"{dataset_num}_{self.predictions[len(self.predictions)-1]}_{f.split('_')[1]}"
                print("new file name ", file_name)
                copyfile(f"{self.image_path}/{f}", f"{self.image_dataset}/{file_name}")



def main(argv):
    """ The main() function. """
    use_stream = False
    use_webcam = False
    try:
        opts, args = getopt.getopt(argv,"psw", ["use_stream", "use_webcam"])
    except getopt.GetoptError:
        print("Useage:")
        print(" -s --use_stream          enable stream")
        print(" -w --use_webcam          use webcam and color tracking")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-s", "--use_stream"):
            use_stream = True
        elif opt in ("-w", "--use_webcam"):
            use_webcam = True
        else:
            print("Useage:")
            print("-s --use_stream          enable stream")
            print("-w --use_webcam          use webcam and color tracking")
            sys.exit(2)

    
    if use_webcam:
        print("Use Webcam and Color Tracking")
        tracker = Tracker(use_web_cam=True, use_colored_pen=True)     # track pen using colored mask and webcam
    elif use_stream:
        print("Enable Stream") 
        tracker = Tracker(enable_stream=True)   # allow streaming to virtual camera
                                                # the default virtual camera address is /dev/video11
                                                # use param `virtual_cam` to specify a different address
    else:
        print("Use Depth Camera and Haar Cascade Object Detection")
        tracker = Tracker()           # default configuration, track pen using haar cascade model


if __name__=="__main__": 
    try:
        main(sys.argv[1:])
    except Exception as inst:
        print(inst)
    finally:
        print("Pipeline End")
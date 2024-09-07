# Standard library imports
import argparse  # For parsing command-line arguments
import json  # For JSON data handling
from pathlib import Path  # For file path operations
import time  # For time-related functions
from typing import Dict, List, Tuple  # For type hinting
import threading  # For multi-threading support
import queue  # For queue data structure

# ByteTrack package for object tracking
from cjm_byte_track.core import BYTETracker
from cjm_byte_track.matching import match_detections_with_tracks

# Utility functions
from cjm_psl_utils.core import download_file  # For downloading files
from cjm_pil_utils.core import resize_img  # For resizing images

# OpenCV for computer vision tasks
import cv2

# NumPy for numerical operations
import numpy as np

# PIL (Python Imaging Library) for image processing
from PIL import Image, ImageDraw, ImageFont

# ONNX (Open Neural Network Exchange) for machine learning interoperability
#import onnx  # Core ONNX functionality
import onnxruntime as ort  # ONNX Runtime for model inference

def load_colormap_data(checkpoint_dir: Path) -> Tuple[List[str], List[Tuple[int, int, int]], Dict[str, Tuple[float, float, float]]]:
    """
    Load and process colormap data from a JSON file in the checkpoint directory.
    
    Args:
    checkpoint_dir (Path): Path to the checkpoint directory containing the colormap JSON file.
    
    Returns:
    Tuple containing:
        - class_names (List[str]): List of class names extracted from the colormap.
        - int_colors (List[Tuple[int, int, int]]): List of RGB color tuples in integer format.
        - colormap_dict (Dict[str, Tuple[float, float, float]]): Dictionary mapping class names to RGB color tuples.
    """
    # Find the colormap JSON file
    colormap_path = list(checkpoint_dir.glob('*colormap.json'))[0]

    # Load the JSON colormap data
    with open(colormap_path, 'r') as file:
        colormap_json = json.load(file)

    # Convert the JSON data to a dictionary
    colormap_dict = {item['label']: tuple(item['color']) for item in colormap_json['items']}

    # Extract the class names from the colormap
    class_names = list(colormap_dict.keys())

    # Make a copy of the colormap in integer format
    int_colors = [tuple(int(c*255) for c in color) for color in colormap_dict.values()]

    return class_names, int_colors, colormap_dict

def prepare_image_for_inference(frame:np.ndarray, target_sz:int, max_stride:int):

    """
    Prepares an image for inference by performing a series of preprocessing steps.
    
    Steps:
    1. Converts a BGR image to RGB.
    2. Resizes the image to a target size without cropping, considering a given divisor.
    3. Calculates input dimensions as multiples of the max stride.
    4. Calculates offsets based on the resized image dimensions and input dimensions.
    5. Computes the scale between the original and resized image.
    6. Crops the resized image based on calculated input dimensions.
    
    Parameters:
    - frame (numpy.ndarray): The input image in BGR format.
    - target_sz (int): The target minimum size for resizing the image.
    - max_stride (int): The maximum stride to be considered for calculating input dimensions.
    
    Returns:
    tuple: 
    - rgb_img (PIL.Image): The converted RGB image.
    - input_dims (list of int): Dimensions of the image that are multiples of max_stride.
    - offsets (numpy.ndarray): Offsets from the resized image dimensions to the input dimensions.
    - min_img_scale (float): Scale factor between the original and resized image.
    - input_img (PIL.Image): Cropped image based on the calculated input dimensions.
    """

    # Convert the BGR image to RGB
    rgb_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Resize image without cropping to multiple of the max stride
    resized_img = resize_img(rgb_img, target_sz=target_sz, divisor=1)
    
    # Calculating the input dimensions that multiples of the max stride
    input_dims = [dim - dim % max_stride for dim in resized_img.size]
    # Calculate the offsets from the resized image dimensions to the input dimensions
    offsets = (np.array(resized_img.size) - input_dims) / 2
    # Calculate the scale between the source image and the resized image
    min_img_scale = min(rgb_img.size) / min(resized_img.size)
    
    # Crop the resized image to the input dimensions
    input_img = resized_img.crop(box=[*offsets, *resized_img.size - offsets])
    
    return rgb_img, input_dims, offsets, min_img_scale, input_img

def generate_output_grids_np(height, width, strides=[8,16,32]):
    """
    Generate a numpy array containing grid coordinates and strides for a given height and width.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        np.ndarray: A numpy array containing grid coordinates and strides.
    """

    all_coordinates = []

    for stride in strides:
        # Calculate the grid height and width
        grid_height = height // stride
        grid_width = width // stride

        # Generate grid coordinates
        g1, g0 = np.meshgrid(np.arange(grid_height), np.arange(grid_width), indexing='ij')

        # Create an array of strides
        s = np.full((grid_height, grid_width), stride)

        # Stack the coordinates along with the stride
        coordinates = np.stack((g0.flatten(), g1.flatten(), s.flatten()), axis=-1)

        # Append to the list
        all_coordinates.append(coordinates)

    # Concatenate all arrays in the list along the first dimension
    output_grids = np.concatenate(all_coordinates, axis=0)

    return output_grids

def calculate_boxes_and_probs(model_output:np.ndarray, output_grids:np.ndarray) -> np.ndarray:
    """
    Calculate the bounding boxes and their probabilities.

    Parameters:
    model_output (numpy.ndarray): The output of the model.
    output_grids (numpy.ndarray): The output grids.

    Returns:
    numpy.ndarray: The array containing the bounding box coordinates, class labels, and maximum probabilities.
    """
    # Calculate the bounding box coordinates
    box_centroids = (model_output[..., :2] + output_grids[..., :2]) * output_grids[..., 2:]
    box_sizes = np.exp(model_output[..., 2:4]) * output_grids[..., 2:]

    x0, y0 = [t.squeeze(axis=2) for t in np.split(box_centroids - box_sizes / 2, 2, axis=2)]
    w, h = [t.squeeze(axis=2) for t in np.split(box_sizes, 2, axis=2)]

    # Calculate the probabilities for each class
    box_objectness = model_output[..., 4]
    box_cls_scores = model_output[..., 5:]
    box_probs = np.expand_dims(box_objectness, -1) * box_cls_scores

    # Get the maximum probability and corresponding class for each proposal
    max_probs = np.max(box_probs, axis=-1)
    labels = np.argmax(box_probs, axis=-1)

    return np.array([x0, y0, w, h, labels, max_probs]).transpose((1, 2, 0))

def process_outputs(outputs:np.ndarray, input_dims:tuple, bbox_conf_thresh:float):

    """
    Process the model outputs to generate bounding box proposals filtered by confidence threshold.
    
    Parameters:
    - outputs (numpy.ndarray): The raw output from the model, which will be processed to calculate boxes and probabilities.
    - input_dims (tuple of int): Dimensions (height, width) of the input image to the model.
    - bbox_conf_thresh (float): Threshold for the bounding box confidence/probability. Bounding boxes with a confidence
                                score below this threshold will be discarded.
    
    Returns:
    - numpy.array: An array of proposals where each proposal is an array containing bounding box coordinates
                   and its associated probability, sorted in descending order by probability.
    """

    # Process the model output
    outputs = calculate_boxes_and_probs(outputs, generate_output_grids_np(*input_dims))
    # Filter the proposals based on the confidence threshold
    max_probs = outputs[:, :, -1]
    mask = max_probs > bbox_conf_thresh
    proposals = outputs[mask]
    # Sort the proposals by probability in descending order
    proposals = proposals[proposals[..., -1].argsort()][::-1]
    return proposals

def calc_iou(proposals:np.ndarray) -> np.ndarray:
    """
    Calculates the Intersection over Union (IoU) for all pairs of bounding boxes (x,y,w,h) in 'proposals'.

    The IoU is a measure of overlap between two bounding boxes. It is calculated as the area of
    intersection divided by the area of union of the two boxes.

    Parameters:
    proposals (2D np.array): A NumPy array of bounding boxes, where each box is an array [x, y, width, height].

    Returns:
    iou (2D np.array): The IoU matrix where each element i,j represents the IoU of boxes i and j.
    """

    # Calculate coordinates for the intersection rectangles
    x1 = np.maximum(proposals[:, 0], proposals[:, 0][:, None])
    y1 = np.maximum(proposals[:, 1], proposals[:, 1][:, None])
    x2 = np.minimum(proposals[:, 0] + proposals[:, 2], (proposals[:, 0] + proposals[:, 2])[:, None])
    y2 = np.minimum(proposals[:, 1] + proposals[:, 3], (proposals[:, 1] + proposals[:, 3])[:, None])
    
    # Calculate intersection areas
    intersections = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    # Calculate union areas
    areas = proposals[:, 2] * proposals[:, 3]
    unions = areas[:, None] + areas - intersections

    # Calculate IoUs
    iou = intersections / unions

    # Return the iou matrix
    return iou

def nms_sorted_boxes(iou:np.ndarray, iou_thresh:float=0.45) -> np.ndarray:
    """
    Applies non-maximum suppression (NMS) to sorted bounding boxes.

    It suppresses boxes that have high overlap (as defined by the IoU threshold) with a box that 
    has a higher score.

    Parameters:
    iou (np.ndarray): An IoU matrix where each element i,j represents the IoU of boxes i and j.
    iou_thresh (float): The IoU threshold for suppression. Boxes with IoU > iou_thresh are suppressed.

    Returns:
    keep (np.ndarray): The indices of the boxes to keep after applying NMS.
    """

    # Create a boolean mask to keep track of boxes
    mask = np.ones(iou.shape[0], dtype=bool)

    # Apply non-max suppression
    for i in range(iou.shape[0]):
        if mask[i]:
            # Suppress boxes with higher index and IoU > threshold
            mask[(iou[i] > iou_thresh) & (np.arange(iou.shape[0]) > i)] = False

    # Return the indices of the boxes to keep
    return np.arange(iou.shape[0])[mask]

def draw_bboxes_pil(image, boxes, labels, colors, font, width=2, font_size=18, probs=None):
    """
    Annotates an image with bounding boxes, labels, and optional probability scores.

    Parameters:
    - image (PIL.Image): The input image on which annotations will be drawn.
    - boxes (list of tuples): A list of bounding box coordinates where each tuple is (x, y, w, h).
    - labels (list of str): A list of labels corresponding to each bounding box.
    - colors (list of str): A list of colors for each bounding box and its corresponding label.
    - font (str): Path to the font file to be used for displaying the labels.
    - width (int, optional): Width of the bounding box lines. Defaults to 2.
    - font_size (int, optional): Size of the font for the labels. Defaults to 18.
    - probs (list of float, optional): A list of probability scores corresponding to each label. Defaults to None.

    Returns:
    - annotated_image (PIL.Image): The image annotated with bounding boxes, labels, and optional probability scores.
    """
    
    # Define a reference diagonal
    REFERENCE_DIAGONAL = 1000
    
    # Scale the font size using the hypotenuse of the image
    font_size = int(font_size * (np.hypot(*image.size) / REFERENCE_DIAGONAL))
    
    # Add probability scores to labels if provided
    if probs is not None:
        labels = [f"{label}: {prob*100:.2f}%" for label, prob in zip(labels, probs)]

    # Create an ImageDraw object for drawing on the image
    draw = ImageDraw.Draw(image)

    # Load the font file (outside the loop)
    fnt = ImageFont.truetype(font, font_size)
    
    # Compute the mean color value for each color
    mean_colors = [np.mean(np.array(color)) for color in colors]

    # Loop through the bounding boxes, labels, and colors
    for box, label, color, mean_color in zip(boxes, labels, colors, mean_colors):
        # Get the bounding box coordinates
        x, y, w, h = box

        # Draw the bounding box on the image
        draw.rectangle([x, y, x+w, y+h], outline=color, width=width)
        
        # Get the size of the label text box
        label_w, label_h = draw.textbbox(xy=(0,0), text=label, font=fnt)[2:]
        
        # Draw the label rectangle on the image
        draw.rectangle([x, y-label_h, x+label_w, y], outline=color, fill=color)

        # Draw the label text on the image
        font_color = 'black' if mean_color > 127.5 else 'white'
        draw.multiline_text((x, y-label_h), label, font=fnt, fill=font_color)
        
    return image

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    """
    Generate a GStreamer pipeline string for capturing and processing video from a camera.

    This function creates a pipeline that captures video from an NVIDIA Argus camera,
    performs necessary conversions, and prepares the video for display or further processing.

    Args:
        sensor_id (int): The ID of the camera sensor to use (default: 0).
        capture_width (int): The width of the captured video in pixels (default: 1920).
        capture_height (int): The height of the captured video in pixels (default: 1080).
        display_width (int): The width of the displayed/processed video in pixels (default: 960).
        display_height (int): The height of the displayed/processed video in pixels (default: 540).
        framerate (int): The desired framerate of the video capture (default: 30).
        flip_method (int): The method used to flip the image, if needed (default: 0, no flip).

    Returns:
        str: A GStreamer pipeline string that can be used with GStreamer-based applications.
    """
    return (
        # Start with nvarguscamerasrc to capture from NVIDIA Argus camera
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        # Set the captured video format and properties
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        # Use nvvidconv to convert the video and potentially flip the image
        f"nvvidconv flip-method={flip_method} ! "
        # Set the display/processing video format and properties
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        # Convert the video color format
        f"videoconvert ! "
        # Set the final video format to BGR for compatibility with OpenCV
        f"video/x-raw, format=(string)BGR ! appsink"
    )

class FrameDropper:
    """
    A class for efficiently reading frames from a video capture device,
    dropping frames if necessary to maintain real-time processing.
    """

    def __init__(self, cv2_capture: cv2.VideoCapture, queue_size=1):
        """
        Initialize the FrameDropper.

        Args:
            cv2_capture (cv2.VideoCapture): The video capture object.
            queue_size (int): Maximum number of frames to keep in the queue.
        """
        self.cap = cv2_capture
        self.q = queue.Queue(maxsize=queue_size)
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        """
        Continuously read frames from the video capture device and manage the frame queue.
        Runs in a separate thread.
        """
        while not self.stop_flag.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.full():
                self.q.put(frame)
            else:
                try:
                    # Discard oldest frame if queue is full
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                self.q.put(frame)

    def read(self):
        """
        Read a frame from the queue.

        Returns:
            tuple: (True, frame) where frame is the next available frame.
        """
        return True, self.q.get()

    def release(self):
        """
        Stop the reading thread and release the video capture resources.
        """
        self.stop_flag.set()
        self.cap.release()
        self.thread.join()

def create_onnx_session(checkpoint_dir: Path) -> ort.InferenceSession:
    # Get the filename for the ONNX model
    # Assumes there's only one .onnx file in the checkpoint directory
    onnx_file_path = list(checkpoint_dir.glob('*.onnx'))[0]

    # Set up a directory for TensorRT engine cache
    trt_cache_dir = checkpoint_dir / 'trt_engine_cache'

    # Initialize ONNX Runtime session options
    sess_opt = ort.SessionOptions()
    # Disable memory optimizations to potentially improve performance
    sess_opt.enable_cpu_mem_arena = False
    sess_opt.enable_mem_pattern = False
    sess_opt.enable_mem_reuse = False
    # Set execution mode to sequential for predictable behavior
    sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Enable all graph optimizations
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Configure TensorRT Execution Provider settings
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,  # GPU device ID (0 for the first GPU)
            'trt_int8_enable': True,  # Enable INT8 precision mode
            'trt_engine_cache_enable': True,  # Enable caching of TensorRT engines
            'trt_engine_cache_path': str(trt_cache_dir),  # Path to store TensorRT cache
            'trt_int8_calibration_table_name': 'calibration.flatbuffers',  # INT8 calibration file
            'trt_max_workspace_size': 4e9,  # Maximum TensorRT workspace size (4GB)
            'trt_timing_cache_enable': True,  # Enable timing cache for faster engine building
            'trt_force_sequential_engine_build': True,  # Build engines sequentially
            'trt_dla_enable': False,  # Disable DLA (Deep Learning Accelerator)
            'trt_max_partition_iterations': 1000,  # Max iterations for partitioning
            'trt_min_subgraph_size': 1,  # Minimum subgraph size for TensorRT
        })
    ]

    # Create an ONNX Runtime InferenceSession with the specified options and providers
    session = ort.InferenceSession(str(onnx_file_path), sess_options=sess_opt, providers=providers)
    
    return session

def setup_camera_capture(use_csi: bool, sensor_id: int, capture_width: int, capture_height: int, 
                         framerate: int, flip_method: int = 0) -> tuple:
    """
    Set up camera capture based on the specified parameters.
    
    Args:
    use_csi (bool): Flag to use CSI camera or V4L2 camera
    sensor_id (int): Camera sensor ID
    capture_width (int): Width of the capture frame
    capture_height (int): Height of the capture frame
    framerate (int): Desired frame rate
    flip_method (int): Flip method for CSI camera (default: 0)
    
    Returns:
    FrameDropper: video_capture
    """

    # Configure camera source based on the 'use_csi' flag
    if use_csi:
        # Use CSI camera with GStreamer pipeline
        src = gstreamer_pipeline(sensor_id=sensor_id,
                           display_width=capture_width,
                           display_height=capture_height,
                           flip_method=flip_method, 
                           capture_width=capture_width, 
                           capture_height=capture_height, 
                           framerate=framerate)
        cv2_capture = cv2.VideoCapture(src)
    else:
        # Use V4L2 camera
        cv2_capture = cv2.VideoCapture(sensor_id, cv2.CAP_V4L2)
        cv2_capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        cv2_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        cv2_capture.set(cv2.CAP_PROP_FPS, framerate)

    # Create a FrameDropper object to handle video capture
    video_capture = FrameDropper(cv2_capture)
    
    return video_capture

def process_frame(frame, session, tracker, class_names, test_sz, max_stride, input_dim_slice, bbox_conf_thresh, iou_thresh, font_file, int_colors):
    # Prepare the input image for inference
    rgb_img, input_dims, offsets, min_img_scale, input_img = prepare_image_for_inference(frame, test_sz, max_stride)
    
    # Convert the input image to NumPy format for the model
    input_tensor_np = np.array(input_img, dtype=np.float32).transpose((2, 0, 1))[None]/255
                    
    # Run inference using the ONNX session
    outputs = session.run(None, {"input": input_tensor_np})[0]

    # Process the model output to get object proposals
    proposals = process_outputs(outputs, input_tensor_np.shape[input_dim_slice], bbox_conf_thresh)
    
    # Apply non-max suppression to filter overlapping proposals
    proposal_indices = nms_sorted_boxes(calc_iou(proposals[:, :-2]), iou_thresh)
    proposals = proposals[proposal_indices]
    
    # Extract bounding boxes, labels, and probabilities from proposals
    bbox_list = (proposals[:,:4]+[*offsets, 0, 0])*min_img_scale
    label_list = [class_names[int(idx)] for idx in proposals[:,4]]
    probs_list = proposals[:,5]

    # Initialize track IDs for detected objects
    track_ids = [-1]*len(bbox_list)

    # Convert bounding boxes to top-left bottom-right (tlbr) format
    tlbr_boxes = bbox_list.copy()
    tlbr_boxes[:, 2:4] += tlbr_boxes[:, :2]

    # Update tracker with detections
    tracks = tracker.update(
        output_results=np.concatenate([tlbr_boxes, probs_list[:, np.newaxis]], axis=1),
        img_info=rgb_img.size,
        img_size=rgb_img.size)

    if len(tlbr_boxes) > 0 and len(tracks) > 0:
        # Match detections with tracks
        track_ids = match_detections_with_tracks(tlbr_boxes=tlbr_boxes, track_ids=track_ids, tracks=tracks)

        # Filter object detections based on tracking results
        bbox_list, label_list, probs_list, track_ids = zip(*[(bbox, label, prob, track_id) 
                                                            for bbox, label, prob, track_id 
                                                            in zip(bbox_list, label_list, probs_list, track_ids) if track_id != -1])
        
        if len(bbox_list) > 0:
            # Annotate the current frame with bounding boxes and tracking IDs
            annotated_img = draw_bboxes_pil(
                image=rgb_img, 
                boxes=bbox_list, 
                labels=[f"{track_id}-{label}" for track_id, label in zip(track_ids, label_list)],
                probs=probs_list,
                colors=[int_colors[class_names.index(i)] for i in label_list],  
                font=font_file,
            )
            annotated_frame = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
            return annotated_frame
    
    return frame

def main_processing_loop(video_capture, session, tracker, class_names, test_sz, max_stride, input_dim_slice, bbox_conf_thresh, iou_thresh, font_file, int_colors):
    window_title = "Camera Feed - Press 'q' to Quit"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            start_time = time.perf_counter()
            
            ret_val, frame = video_capture.read()
            
            if not ret_val:
                print("Failed to retrieve frame")
                continue
        
            annotated_frame = process_frame(frame, session, tracker, class_names, test_sz, max_stride, input_dim_slice, bbox_conf_thresh, iou_thresh, font_file, int_colors)
        
            # Calculate and display FPS
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            fps = 1 / processing_time
        
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the annotated frame
            cv2.imshow(window_title, annotated_frame)
            
            # Check for 'q' key press to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Clean up resources
        video_capture.release()
        cv2.destroyAllWindows()

def main(checkpoint_dir:Path, camera_type:str, sensor_id:int):

    # Set the name of the font file
    font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'

    # Download the font file
    download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")

    class_names, int_colors, colormap_dict = load_colormap_data(checkpoint_dir)

    # Define Camera Feed Settings
    use_csi = 'csi' in camera_type
    flip_method = 0
    framerate=60
    capture_width=1280
    capture_height=720
    
    # Define Inference Parameters
    test_sz = 384
    bbox_conf_thresh = 0.35
    iou_thresh = 0.45

    # Set the Preprocessing and Post-Processing Parameters
    max_stride = 32
    input_dim_slice = slice(2, 4, None)
    
    session = create_onnx_session(checkpoint_dir)

    video_capture = setup_camera_capture(use_csi, 
                                         sensor_id, 
                                         capture_width, 
                                         capture_height, 
                                         framerate, 
                                         flip_method=0)

    tracker = BYTETracker(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

    main_processing_loop(video_capture, 
                         session, 
                         tracker, 
                         class_names, 
                         test_sz, 
                         max_stride, 
                         input_dim_slice, 
                         bbox_conf_thresh, 
                         iou_thresh, 
                         font_file, 
                         int_colors
                         )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Object-Tracking Demo")
    parser.add_argument("--checkpoint_dir", type=str,
                    help="The path to the the model checkpoint folder.")
    parser.add_argument("--camera_type", type=str, choices=['csi', 'usb'],
                    help="The camera type.")
    parser.add_argument("--sensor_id", type=int, default=0,
                    help="The index for the target camera.")
    args = parser.parse_args()

    main(Path(args.checkpoint_dir), args.camera_type, args.sensor_id)

"""
Object Detection and Tracking Script with Optional File Paths
This script captures frames from a camera, runs object detection using a Hailo device,
tracks the detected objects using ByteTrack, and displays the annotated frames with bounding boxes.

Users can optionally provide file paths for the HEF model and JSON colormap files.
If not provided, the script will search for these files in the current directory.

Usage:
    python yolox-hailo-bytetrack-rpi.py [--hef_path HEF_PATH] [--colormap_path COLORMAP_PATH]

Press 'q' to quit the script.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

# Import required packages
try:
    from cjm_psl_utils.core import download_file
except ImportError:
    print("Error: cjm_psl_utils module not found. Please install it before running this script.")
    sys.exit(1)

try:
    from cjm_byte_track.core import BYTETracker
    from cjm_byte_track.matching import match_detections_with_tracks
except ImportError:
    print("Error: cjm_byte_track module not found. Please install it before running this script.")
    sys.exit(1)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from hailo_platform import (
        HEF,
        ConfigureParams,
        FormatType,
        HailoSchedulingAlgorithm,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )
except ImportError:
    print("Error: hailo_platform module not found. Please install the Hailo SDK before running this script.")
    sys.exit(1)

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 module not found. Please install it before running this script.")
    sys.exit(1)


class ImageTransformData(NamedTuple):
    """
    A data class that stores transformation information applied to an image.

    Attributes:
        offset (Tuple[int, int]): The (x, y) offset where the resized image was pasted.
        scale (float): The scaling factor applied to the original image.
    """
    offset: Tuple[int, int]
    scale: float


def resize_and_pad(
    image: np.ndarray,
    target_sz: Tuple[int, int],
    return_transform_data: bool = False,
    fill_color: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[np.ndarray, Optional[ImageTransformData]]:
    """
    Resize an image while maintaining its aspect ratio and pad it to fit the target size.

    Args:
        image (np.ndarray): The original image as a numpy array.
        target_sz (Tuple[int, int]): The desired size (width, height) for the output image.
        return_transform_data (bool, optional): If True, returns transformation data (offset and scale).
        fill_color (Tuple[int, int, int], optional): The color to use for padding (default is white).

    Returns:
        Tuple[np.ndarray, Optional[ImageTransformData]]: The resized and padded image,
        and optionally the transformation data.
    """
    target_width, target_height = target_sz
    orig_height, orig_width = image.shape[:2]

    aspect_ratio = orig_width / orig_height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
        scale = target_width / orig_width
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
        scale = target_height / orig_height

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    padded_image = np.full((target_height, target_width, 3), fill_color, dtype=np.uint8)
    padded_image[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = resized_image

    if return_transform_data:
        transform_data = ImageTransformData(offset=(paste_x, paste_y), scale=scale)
        return padded_image, transform_data
    else:
        return padded_image, None


def adjust_bbox(
    bbox: Tuple[float, float, float, float],
    transform_data: ImageTransformData
) -> Tuple[float, float, float, float]:
    """
    Adjust a bounding box according to the transformation data (offset and scale).

    Args:
        bbox (Tuple[float, float, float, float]): The original bounding box as (x, y, width, height).
        transform_data (ImageTransformData): The transformation data containing offset and scale.

    Returns:
        Tuple[float, float, float, float]: The adjusted bounding box.
    """
    x, y, w, h = bbox
    offset_x, offset_y = transform_data.offset
    scale = transform_data.scale

    adjusted_x = (x - offset_x) / scale
    adjusted_y = (y - offset_y) / scale
    adjusted_w = w / scale
    adjusted_h = h / scale

    return (adjusted_x, adjusted_y, adjusted_w, adjusted_h)


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


def draw_bboxes_pil(image, boxes, labels, colors, font, width:int=2, font_size:int=18, probs=None):
    """
    Annotates an image with bounding boxes, labels, and optional probability scores.

    This function draws bounding boxes on the provided image using the given box coordinates,
    colors, and labels. If probabilities are provided, they will be added to the labels.

    Parameters:
    image (PIL.Image): The input image on which annotations will be drawn.
    boxes (list of tuples): A list of bounding box coordinates where each tuple is (x, y, w, h).
    labels (list of str): A list of labels corresponding to each bounding box.
    colors (list of str): A list of colors for each bounding box and its corresponding label.
    font (str): Path to the font file to be used for displaying the labels.
    width (int, optional): Width of the bounding box lines. Defaults to 2.
    font_size (int, optional): Size of the font for the labels. Defaults to 18.
    probs (list of float, optional): A list of probability scores corresponding to each label. Defaults to None.

    Returns:
    annotated_image (PIL.Image): The image annotated with bounding boxes, labels, and optional probability scores.
    """

    # Define a reference diagonal
    REFERENCE_DIAGONAL = 1000

    # Scale the font size using the hypotenuse of the image
    font_size = int(font_size * (np.hypot(*image.size) / REFERENCE_DIAGONAL))

    # Add probability scores to labels
    if probs is not None:
        labels = [f"{label}: {prob*100:.2f}%" for label, prob in zip(labels, probs)]

    # Create a copy of the image
    annotated_image = image.copy()

    # Create an ImageDraw object for drawing on the image
    draw = ImageDraw.Draw(annotated_image)

    # Loop through the bounding boxes and labels
    for i in range(len(labels)):
        # Get the bounding box coordinates
        x, y, w, h = boxes[i]

        # Create a tuple of coordinates for the bounding box
        shape = (x, y, x+w, y+h)

        # Draw the bounding box on the image
        draw.rectangle(shape, outline=colors[i], width=width)

        # Load the font file
        fnt = ImageFont.truetype(font, font_size)

        # Draw the label box on the image
        label_w, label_h = draw.textbbox(xy=(0,0), text=labels[i], font=fnt)[2:]
        draw.rectangle((x, y-label_h, x+label_w, y), outline=colors[i], fill=colors[i], width=width)

        # Draw the label on the image
        draw.multiline_text((x, y-label_h), labels[i], font=fnt, fill='black' if np.mean(colors[i]) > 127.5 else 'white')

    return annotated_image


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Object Detection and Tracking Script")
    parser.add_argument('--hef_path', type=str, help='Path to the HEF model file')
    parser.add_argument('--colormap_path', type=str, help='Path to the JSON colormap file')
    args = parser.parse_args()

    # Set the path to the checkpoint directory
    checkpoint_dir = Path("./")

    # Check if the font file exists, if not, download it
    font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'
    font_path = checkpoint_dir / font_file
    if not font_path.exists():
        print("Downloading font file...")
        download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", str(font_path.parent))
    else:
        print("Font file already exists.")

    # Load the colormap
    if args.colormap_path:
        colormap_path = Path(args.colormap_path)
        if not colormap_path.exists():
            print(f"Error: Colormap file '{colormap_path}' does not exist.")
            sys.exit(1)
    else:
        colormap_files = list(checkpoint_dir.glob('*colormap.json'))
        if not colormap_files:
            print("Error: Colormap JSON file not found in the checkpoint directory.")
            sys.exit(1)
        colormap_path = colormap_files[0]

    with open(colormap_path, 'r') as file:
        colormap_json = json.load(file)
    colormap_dict = {item['label']: item['color'] for item in colormap_json['items']}
    class_names = list(colormap_dict.keys())
    int_colors = [tuple(int(c*255) for c in color) for color in colormap_dict.values()]

    # Load the compiled HEF model
    if args.hef_path:
        hef_file_path = Path(args.hef_path)
        if not hef_file_path.exists():
            print(f"Error: HEF model file '{hef_file_path}' does not exist.")
            sys.exit(1)
    else:
        hef_files = list(checkpoint_dir.glob('*.hef'))
        if not hef_files:
            print("Error: HEF model file not found in the checkpoint directory.")
            sys.exit(1)
        hef_file_path = hef_files[0]

    print(f"Loading HEF model from {hef_file_path}...")
    hef = HEF(str(hef_file_path))

    # Set VDevice params to disable the HailoRT service feature
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE

    # Create a Hailo virtual device with the specified parameters
    target = VDevice(params=params)

    # Configure the device with the HEF and PCIe interface
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)

    # Select the first network group (there's only one in this case)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()

    # Create input and output virtual streams params
    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    # Get information about the input and output virtual streams
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]

    # Set inference parameters
    # Set the desired preview size
    preview_width, preview_height = 960, 540
    bbox_conf_thresh = 0.35
    iou_thresh = 0.45

    # Set up window title for display
    window_title = "Camera Feed - Press 'q' to Quit"

    # Create a Picamera2 object
    picam2 = Picamera2()

    # Print available sensor modes
    print("Available sensor modes:")
    for i, mode in enumerate(picam2.sensor_modes):
        print(f"Mode {i}: {mode['size']} at {mode['fps']}fps")

    # Choose a mode (using the first mode)
    chosen_mode = picam2.sensor_modes[0]

    # Create a configuration using the chosen mode
    config = picam2.create_preview_configuration(
        main={
            "size": chosen_mode["size"],
            "format": "RGB888"
        },
        controls={
            "FrameRate": chosen_mode["fps"]
        },
        sensor={
            "output_size": chosen_mode["size"],
            "bit_depth": chosen_mode["bit_depth"]
        }
    )

    # Configure the camera
    picam2.configure(config)

    # Start the camera
    picam2.start()

    # Initialize the ByteTracker for object tracking
    tracker = BYTETracker(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

    try:
        # Main processing loop
        while True:
            start_time = time.perf_counter()

            # Capture a frame
            frame = picam2.capture_array()

            # Resize and pad the image to the model's input size
            input_img_np, transform_data = resize_and_pad(frame, input_vstream_info.shape[::-1][1:], True)

            # Preprocess the image
            input_tensor_np = np.array(input_img_np, dtype=np.float32)[None]/255
            input_tensor_np = np.ascontiguousarray(input_tensor_np)

            # Run inference
            input_data = {input_vstream_info.name: input_tensor_np}
            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                with network_group.activate(network_group_params):
                    infer_results = infer_pipeline.infer(input_data)

            # Process the model output
            outputs = infer_results[output_vstream_info.name].transpose(0, 1, 3, 2)[0]

            # Process outputs to get proposals
            proposals = process_outputs(outputs, input_vstream_info.shape[:-1], bbox_conf_thresh)

            # Apply non-max suppression
            proposal_indices = nms_sorted_boxes(calc_iou(proposals[:, :-2]), iou_thresh)
            proposals = proposals[proposal_indices]

            # Extract bounding boxes, labels, and probabilities
            bbox_list = [adjust_bbox(bbox, transform_data) for bbox in proposals[:, :4]]
            label_list = [class_names[int(idx)] for idx in proposals[:, 4]]
            probs_list = proposals[:, 5]

            # Initialize track IDs
            track_ids = [-1]*len(bbox_list)

            # Convert bounding boxes to tlbr format
            tlbr_boxes = np.array(bbox_list).reshape(-1, 4).copy()
            tlbr_boxes[:, 2:4] += tlbr_boxes[:, :2]

            # Update tracker
            tracks = tracker.update(
                output_results=np.concatenate([tlbr_boxes, probs_list[:, np.newaxis]], axis=1),
                img_info=frame.shape[::-1][1:],
                img_size=frame.shape[::-1][1:]
            )

            if len(tlbr_boxes) > 0 and len(tracks) > 0:
                # Match detections with tracks
                track_ids = match_detections_with_tracks(tlbr_boxes=tlbr_boxes, track_ids=track_ids, tracks=tracks)

                # Filter detections based on tracking results
                detections = [(bbox, label, prob, track_id) for bbox, label, prob, track_id in
                              zip(bbox_list, label_list, probs_list, track_ids) if track_id != -1]

                if detections:
                    bbox_list, label_list, probs_list, track_ids = zip(*detections)

                    # Annotate the frame
                    annotated_img = draw_bboxes_pil(
                        image=Image.fromarray(frame),
                        boxes=bbox_list,
                        labels=[f"{track_id}-{label}" for track_id, label in zip(track_ids, label_list)],
                        probs=probs_list,
                        colors=[int_colors[class_names.index(i)] for i in label_list],
                        font=str(font_path),
                    )
                    annotated_frame = cv2.resize(np.array(annotated_img), (preview_width, preview_height))
                else:
                    # If no detections, use the original frame
                    annotated_frame = cv2.resize(frame, (preview_width, preview_height))
            else:
                # If no detections, use the original frame
                annotated_frame = cv2.resize(frame, (preview_width, preview_height))

            # Calculate and display FPS
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            fps = 1 / processing_time if processing_time > 0 else 0

            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the annotated frame
            cv2.imshow(window_title, annotated_frame)

            # Check for 'q' key press to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop the camera and close the preview window
        cv2.destroyAllWindows()
        picam2.close()
        target.release()  # Release the Hailo device


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
'''
    July 2023wbk.  Turned the object_detection_demo_yolov3_async.py sample program from
    OpenVINO-YOLOV4-master github code into this thread function making minimal changes.
    This used som OpenVINO features that I do not understand and couldn't easily find
    documentation for, but treating these code sections as a "black box" worked fine.
    Maybe someday I'll dive in and see if understanding it would let me improve effeciency.
    But is seems more than good enough for now.
'''

"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging
import threading
import os
import sys
from collections import deque
#from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import perf_counter
from enum import Enum

import cv2
import numpy as np
from openvino.inference_engine import IECore

import ngraph as ng
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()
'''
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      default="fp16/frozen_darknet_yolov4_model.xml", type=str)
    args.add_argument("-i", "--input", help="Path to an image/video file. (Specify 'cam' to work with camera)",
                      default="rtsp://admin:admin@192.168.2.162:554/media/video1", type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is GPU", default="GPU", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default="cfg/coco.names", type=str)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    args.add_argument("-nireq", "--num_infer_requests", help="Optional. Number of infer requests",
                      default=1, type=int)
    args.add_argument("-nstreams", "--num_streams",
                      help="Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode "
                           "(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> "
                           "or just <nstreams>)",
                      default="", type=str)
    args.add_argument("-nthreads", "--number_threads",
                      help="Optional. Number of threads to use for inference on CPU (including HETERO cases)",
                      default=None, type=int)
    args.add_argument("-loop_input", "--loop_input", help="Optional. Iterate over input infinitely",
                      action='store_true')
    args.add_argument("-no_show", "--no_show", help="Optional. Don't show output", action='store_true')
    args.add_argument("--keep_aspect_ratio", action="store_true", default=False,
                      help='Optional. Keeps aspect ratio on resize.')
    return parser
'''

global __Thread__
__Thread__ = False

global __verifyConf__
__verifyConf__ = 0.73

global __device__
__device__ = "GPU"

# setup default parameters, since it is not called from a command line
# I'm keeping the sample code as unchanged as possible to make the yolo verification thread
# Until I can prove it will work on the iGPU it is not worth figuring out the details of how it works.
args_num_infer_requests = 1
args_number_threads = None
args_num_streams =""
args_keep_aspect_ratio = False
args_cpu_extension = None
args_labels = "cfg/coco.names"
args_raw_output_message = False
args_iou_threshold =0.4
args_no_show = False
### default to use GPU but accept -myriad option to use NCS2 in stead of GPU.
### Might be marginally useful with Atomic Pi or Pi3B etc.
#args_device = "MYRIAD"  # works with NCS2 but only ~2 fps on demo that gave ~6fps with GPU
#args_device = "GPU"


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else param['anchors']

        self.isYoloV3 = False

        if param.get('mask'):
            mask = param['mask']
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True # Weak way to determine but the only one.


class Modes(Enum):
    USER_SPECIFIED = 0
    MIN_LATENCY = 1


class Mode():
    def __init__(self, value):
        self.current = value

    def next(self):
        if self.current.value + 1 < len(Modes):
            self.current = Modes(self.current.value + 1)
        else:
            self.current = Modes(0)


class ModeInfo():
    def __init__(self):
        self.last_start_time = perf_counter()
        self.last_end_time = None
        self.frames_count = 0
        self.latency_sum = 0


def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, is_proportional):
    if is_proportional:
        scale = np.array([min(im_w/im_h, 1), min(im_h/im_w, 1)])
        offset = 0.5*(np.ones(2) - scale)
        x, y = (np.array([x, y]) - offset) / scale
        width, height = np.array([width, height]) / scale
    xmin = int((x - width / 2) * im_w)
    ymin = int((y - height / 2) * im_h)
    xmax = int(xmin + width * im_w)
    ymax = int(ymin + height * im_h)
    # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
    # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())


def parse_yolo_region(predictions, resized_image_shape, original_im_shape, params, threshold, is_proportional):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = predictions.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    size_normalizer = (resized_image_w, resized_image_h) if params.isYoloV3 else (params.side, params.side)
    bbox_size = params.coords + 1 + params.classes
    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for row, col, n in np.ndindex(params.side, params.side, params.num):
        # Getting raw values for each detection bounding box
        bbox = predictions[0, n*bbox_size:(n+1)*bbox_size, row, col]
        x, y, width, height, object_probability = bbox[:5]
        class_probabilities = bbox[5:]
        if object_probability < threshold:
            continue
        # Process raw value
        x = (col + x) / params.side
        y = (row + y) / params.side
        # Value for exp is very big number in some cases so following construction is using here
        try:
            width = exp(width)
            height = exp(height)
        except OverflowError:
            continue
        # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
        width = width * params.anchors[2 * n] / size_normalizer[0]
        height = height * params.anchors[2 * n + 1] / size_normalizer[1]

        class_id = np.argmax(class_probabilities)
        confidence = class_probabilities[class_id]*object_probability
        if confidence < threshold:
            continue
        objects.append(scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
                                  im_h=orig_im_h, im_w=orig_im_w, is_proportional=is_proportional))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def resize(image, size, keep_aspect_ratio, interpolation=cv2.INTER_LINEAR):
    if not keep_aspect_ratio:
        return cv2.resize(image, size, interpolation=interpolation)

    iw, ih = image.shape[0:2][::-1]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw, nh), interpolation=interpolation)
    new_image = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    dx = (w-nw)//2
    dy = (h-nh)//2
    new_image[dy:dy+nh, dx:dx+nw, :] = image
    return new_image


def preprocess_frame(frame, input_height, input_width, nchw_shape, keep_aspect_ratio):
    in_frame = resize(frame, (input_width, input_height), keep_aspect_ratio)
    if nchw_shape:
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = np.expand_dims(in_frame, axis=0)
    return in_frame


def get_objects(output, net, new_frame_height_width, source_height_width, prob_threshold, is_proportional):
    objects = list()
    function = ng.function_from_cnn(net)
    for layer_name, out_blob in output.items():
        # forgot, is this commenting out original author's or mine?
        #out_blob = out_blob.buffer.reshape(net.layers[net.layers[layer_name].parents[0]].out_data[0].shape)
        #layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
        out_blob = out_blob.buffer.reshape(net.outputs[layer_name].shape)
        params = [x._get_attributes() for x in function.get_ordered_ops() if x.get_friendly_name() == layer_name][0]
        layer_params = YoloParams(params, out_blob.shape[2])
        objects += parse_yolo_region(out_blob, new_frame_height_width, source_height_width, layer_params,
                                     prob_threshold, is_proportional)

    return objects


def filter_objects(objects, iou_threshold, prob_threshold):
    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    return tuple(obj for obj in objects if obj['confidence'] >= prob_threshold)


def async_callback(status, callback_args):
    request, frame_id, frame_mode, frame, start_time, completed_request_results, empty_requests, \
    mode, event, callback_exceptions = callback_args

    try:
        if status != 0:
            raise RuntimeError('Infer Request has returned status code {}'.format(status))

        completed_request_results[frame_id] = (frame, request.output_blobs, start_time, frame_mode == mode.current)

        if mode.current == frame_mode:
            empty_requests.append(request)
    except Exception as e:
        callback_exceptions.append(e)

    event.set()


def put_highlighted_text(frame, message, position, font_face, font_scale, color, thickness):
    cv2.putText(frame, message, position, font_face, font_scale, (255, 255, 255), thickness + 1) # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)


def await_requests_completion(requests):
    for request in requests:
        request.wait()

###def main():
    ###args = build_argparser().parse_args()
def yolo4ov_thread(results, yoloQ):
    global __Thread__
    global __verifyConf__
    global __device__
    
    args_device = __device__
    print("Starting OpenVINO Yolo4 " + args_device + " verification thread...")
    if yoloQ is None:
        print(    "ERROR! no yolo Queue!")
    yoloVerified=0
    yoloRejected=0
    yoloWaited = 0
    ecnt=0
    dcnt=0
    ncnt=0

 
    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    log.info("Creating Inference Engine...")
    ie = IECore()

    config_user_specified = {}
    config_min_latency = {}

    devices_nstreams = {}
    if args_num_streams:
        devices_nstreams = {device: args_num_streams for device in ['CPU', 'GPU'] if device in args_device} \
                           if args_num_streams.isdigit() \
                           else dict([device.split(':') for device in args_num_streams.split(',')])
    '''
    if 'CPU' in args_device:
        if args_cpu_extension:
            ie.add_extension(args_cpu_extension, 'CPU')
        if args_number_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(args_number_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                                                              if int(devices_nstreams['CPU']) > 0 \
                                                              else 'CPU_THROUGHPUT_AUTO'

        config_min_latency['CPU_THROUGHPUT_STREAMS'] = '1'
    '''
    if 'GPU' in args_device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                                                              if int(devices_nstreams['GPU']) > 0 \
                                                              else 'GPU_THROUGHPUT_AUTO'

        config_min_latency['GPU_THROUGHPUT_STREAMS'] = '1'

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    log.info("Loading network")
    net = ie.read_network("yolo4_OpenVINO/fp16/frozen_darknet_yolov4_model.xml", "yolo4_OpenVINO/fp16/frozen_darknet_yolov4_model.bin")
    # ---------------------------------- 3. Load CPU extension for support specific layer ------------------------------


    # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
    log.info("Preparing inputs")
    input_blob = next(iter(net.input_info))

    # Read and pre-process input images
    if net.input_info[input_blob].input_data.shape[1] == 3:
        input_height, input_width = net.input_info[input_blob].input_data.shape[2:]
        nchw_shape = True
    else:
        input_height, input_width = net.input_info[input_blob].input_data.shape[1:3]
        nchw_shape = False

    if args_labels:
        with open(args_labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    ###input_stream = 0 if args.input == "cam" else args.input

    mode = Mode(Modes.USER_SPECIFIED)
    ###cap = cv2.VideoCapture(input_stream)
    wait_key_time = 1

    # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
    log.info("Loading model to the plugin")
    exec_nets = {}

    exec_nets[Modes.USER_SPECIFIED] = ie.load_network(network=net, device_name=args_device,
                                                      config=config_user_specified,
                                                      num_requests=args_num_infer_requests)
    exec_nets[Modes.MIN_LATENCY] = ie.load_network(network=net, device_name=args_device.split(":")[-1].split(",")[0],
                                                   config=config_min_latency,
                                                   num_requests=1)

    empty_requests = deque(exec_nets[mode.current].requests)
    completed_request_results = {}
    next_frame_id = 0
    next_frame_id_to_show = 0
    mode_info = { mode.current: ModeInfo() }
    event = threading.Event()
    callback_exceptions = []

    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    '''
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between min_latency/user_specified modes, press TAB key in the output window")
    '''


    print("OpenVINO Yolo4 " + args_device + "  verification thread is running...")
    __Thread__ = True

    while __Thread__ is True and not callback_exceptions:
#  process inference         
        personDetected = False
        verifyConf = 0.0
        if next_frame_id_to_show in completed_request_results:
            frame, output, start_time, is_same_mode = completed_request_results.pop(next_frame_id_to_show)

            next_frame_id_to_show += 1
            if is_same_mode:
                mode_info[mode.current].frames_count += 1

            objects = get_objects(output, net, (input_height, input_width), frame.shape[:-1], __verifyConf__,
                                  args_keep_aspect_ratio)
            objects = filter_objects(objects, args_iou_threshold, __verifyConf__)
            '''
            if len(objects) and args_raw_output_message:
                log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")
            '''
            origin_im_size = frame.shape[:-1]
            for obj in objects:
                if obj['class_id']==0 and obj['confidence'] >= __verifyConf__:
                    personDetected = True
                    verifyConf = max(obj['confidence'], verifyConf)  # seems "async" nature of this code means obj can change after for loop exits.
                # Validation bbox of detected object
                obj['xmax'] = min(obj['xmax'], origin_im_size[1])
                obj['ymax'] = min(obj['ymax'], origin_im_size[0])
                obj['xmin'] = max(obj['xmin'], 0)
                obj['ymin'] = max(obj['ymin'], 0)
                if obj['class_id']==0:
                    color = (50, 255, 50)
                else:
                    color = (min(obj['class_id'] * 12.5, 255),
                         min(obj['class_id'] * 7, 255),
                         min(obj['class_id'] * 5, 255))
                det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else str(obj['class_id'])
                '''
                if args_raw_output_message:
                    log.info(
                        "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'],
                                                                                  obj['xmin'], obj['ymin'], obj['xmax'],
                                                                                  obj['ymax'],
                                                                                  color))
                '''
                cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
                cv2.putText(frame, det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                            (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            '''
            # Draw performance stats over frame
            if mode_info[mode.current].frames_count != 0:
                fps_message = "FPS: {:.1f}".format(mode_info[mode.current].frames_count / \
                                                   (perf_counter() - mode_info[mode.current].last_start_time))
                mode_info[mode.current].latency_sum += perf_counter() - start_time
                latency_message = "Latency: {:.1f} ms".format((mode_info[mode.current].latency_sum / \
                                                              mode_info[mode.current].frames_count) * 1e3)

                put_highlighted_text(frame, fps_message, (15, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
                put_highlighted_text(frame, latency_message, (15, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
            
            mode_message = "{} mode".format(mode.current.name)
            put_highlighted_text(frame, mode_message, (10, int(origin_im_size[0] - 20)),
                                 cv2.FONT_HERSHEY_COMPLEX, 0.75, (10, 10, 200), 2)
            '''
            '''
            if not args_no_show:
                (IH, IW) = frame.shape[:2]
                if IW <= 1280:
                    cv2.imshow("Detection Results", frame)
                else:
                    rw = int(IW/(IW/1280))
                    rh = int(IH/(IW/1280))
                    cv2.imshow("Detection Results", cv2.resize(frame, (rw, rh), interpolation = cv2.INTER_AREA))
                key = cv2.waitKey(wait_key_time)

                if key in {ord("q"), ord("Q"), 27}: # ESC key
                    break
                if key == 9: # Tab key
                    prev_mode = mode.current
                    mode.next()

                    await_requests_completion(exec_nets[prev_mode].requests)
                    empty_requests.clear()
                    empty_requests.extend(exec_nets[mode.current].requests)

                    mode_info[prev_mode].last_end_time = perf_counter()
                    mode_info[mode.current] = ModeInfo()
                    '''
            try:
                # Note these boxpoints are for the persons in the verification image which we ignore here.
                ###image, personDetected, _, detectConfidence = do_inference( yolo_frame, model, __verifyConf__ )
                if personDetected is True:   # yolov4 has verified the MobilenetSSDv2 person detection
                    ## image is the yolo_frame with the yolo detection boxes overlaid (from the boxpoints).
                    ## boxPoints are from the SSD inference.
                    # draw the verification confidence onto the ssd_frame
                    yoloVerified+=1
                    text = "Yolo4ov: {:.1f}%".format(verifyConf * 100)   # show verification confidence on detection image
                    cv2.putText(ssd_frame, text, (2, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    if results.full():
                        [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result
                        dcnt+=1
                    results.put((ssd_frame, cam, True, imageDT, ai, boxPoints, frame.copy()), True, 1.0)
                    ###print(detections, boxpoints)    # lets take a look at what we are getting
                else:
                    yoloRejected+=1
                    if results.full():
                        [_,_,_,_,_,_,_]=results.get(False)  # remove oldest result 
                        ncnt+=1                       
                    results.put((ssd_frame, cam, False, imageDT, ai, (-2,0, 0,0, 0,0, 0,0), frame.copy()), True, 1.0)
            except Exception as e:
                ###print('[Exception] yolo_thread results.put() '+ str(cam) + ': ' + str(e))
                ecnt+=1
                continue

# get next frame
        elif empty_requests: # and cap.isOpened():
            # get next frame to process from yolo queue
            try:
                # ssd_frame is full camera resolution with SSD detection box overlaid
                # yolo_frame is "zoomed in" on the SSD detection box and resized to 608x608 for yolo4 inference
                #yoloQ.put((image, cam, personDetected, imageDT, aiStr, boxPoints, yolo_frame), True, 1.0)
                ssd_frame, cam, personDetected, imageDT, ai, boxPoints, yolo_frame = yoloQ.get(True, 1.0)
                ##print("yoloQ got frame.")
            except Exception as e:
                ###print('[Exception] yoloQ.get(): ' + str(e))
                yoloWaited+=1
                continue

            start_time = perf_counter()
            '''
            ret, frame = cap.read()
            if not ret:
                if args.loop_input:
                    cap.open(input_stream)
                else:
                    cap.release()
                continue
            '''
            request = empty_requests.popleft()

            # resize input_frame to network size
            in_frame = preprocess_frame(yolo_frame, input_height, input_width, nchw_shape, args_keep_aspect_ratio)

            # Start inference
            request.set_completion_callback(py_callback=async_callback,
                                            py_data=(request,
                                                     next_frame_id,
                                                     mode.current,
                                                     yolo_frame,
                                                     start_time,
                                                     completed_request_results,
                                                     empty_requests,
                                                     mode,
                                                     event,
                                                     callback_exceptions))
            request.async_infer(inputs={input_blob: in_frame})
            next_frame_id += 1

        else:
            event.wait()


# while loop exits
    if callback_exceptions:
        raise callback_exceptions[0]
    '''
    for mode_value in mode_info.keys():
        log.info("")
        log.info("Mode: {}".format(mode_value.name))

        end_time = mode_info[mode_value].last_end_time if mode_value in mode_info \
                                                          and mode_info[mode_value].last_end_time is not None \
                                                       else perf_counter()
        log.info("FPS: {:.1f}".format(mode_info[mode_value].frames_count / \
                                      (end_time - mode_info[mode_value].last_start_time)))
        log.info("Latency: {:.1f} ms".format((mode_info[mode_value].latency_sum / \
                                             mode_info[mode_value].frames_count) * 1e3))
    '''
    for exec_net in exec_nets.values():
        await_requests_completion(exec_net.requests)

    print("OpenVINO " + args_device + " Yolo4 frames Verified: {}, Rejected: {},  Waited: {} seconds.".format(str(yoloVerified), str(yoloRejected), str(yoloWaited)))
    print("   Verified dropped: " + str(dcnt) + " results dropped: " + str(ncnt) + " results.put() exceptions: " + str(ecnt))


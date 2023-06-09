from absl.flags import FLAGS
from absl import app, flags
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import tools.utils as utils
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from ultralytics import YOLO

flags.DEFINE_string("reid_weights", "./data/onnx/veriwild_dynamic.onnx", "ReID pretrained model")
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean("plot_tracking", True, "True if don't want tracking non-matched car ")
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_integer('size', 640, 'resize images to')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float("conf", 0.25, "conf threshold")
flags.DEFINE_float('score', 0.5, 'score threshold')
flags.DEFINE_string('video1', './data/video/area_wide.avi', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('video2', './data/video/area_small.avi', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('masking_image1', './data/mask/masking_img2.jpg', 'path to masking image file')
flags.DEFINE_string('masking_image2', './data/mask/masking_img1.jpg', 'path to masking image file')

def main(_argv):
    # Definition of the parameters
    min_cosine_distance = 0.4
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initializing deep sort
    model_filename = FLAGS.reid_weights
    encoder_sort = gdet.create_box_encoder(model_filename, batch_size=1)
    encoder_reid = gdet.create_box_encoder(model_filename)

    # initializing YOLOv8
    yolo = YOLO('./checkpoints/yolov8m.pt')
    
    # calculate cosine distance metric
    metric1 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    
    # initialize tracker
    tracker1 = Tracker(metric1)
    tracker2 = Tracker(metric2)
    tracker = [tracker1, tracker2]

    input_size = FLAGS.size
    video_path1 = FLAGS.video1
    video_path2 = FLAGS.video2

    # begin video capture
    try:
        vid1 = cv2.VideoCapture(int(video_path1))
    except:
        vid1 = cv2.VideoCapture(video_path1)
    try:
        vid2 = cv2.VideoCapture(int(video_path2))
    except:
        vid2 = cv2.VideoCapture(video_path2)

    v1width = int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH))
    v1height = int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v2width = int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH))
    v2height = int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = [v1width, v2width]
    height = [v1height, v2height]

    # get video ready to save locally if flag is set
    if FLAGS.output:
        fps = int(vid1.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (v1width, v1height))

    # initializing color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    frame_num = 0
    match_id = -1
    max_instance = [0, 0]
    out = None

    # while video is running
    while True:
        return_value1, frame1 = vid1.read()
        return_value2, frame2 = vid2.read()

        if return_value1 and return_value2:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_num += 1
        print('Frame #: ', frame_num)

        # video mask
        if FLAGS.masking_image1 and FLAGS.masking_image2:
            mask1 = cv2.imread(FLAGS.masking_image1)//255
            mask2 = cv2.imread(FLAGS.masking_image2)//255
            image_data1 = cv2.resize(frame1 * mask1, (input_size, input_size))
            image_data2 = cv2.resize(frame2 * mask2, (input_size, input_size))
        else:
            image_data1 = cv2.resize(frame1, (input_size, input_size))
            image_data2 = cv2.resize(frame2, (input_size, input_size))

        frame = [frame1, frame2]
        image_data = [image_data1, image_data2]

        for i in range(2):
            # YOLO setting
            results = yolo(image_data[i], iou = FLAGS.iou, conf = FLAGS.conf, verbose=False)[0]
            
            factor = np.array([width[i], height[i], width[i], height[i]])[np.newaxis, ...]
            boxes = results.boxes.xyxyn.cpu().numpy()
            boxes[:, 2:] = results.boxes.xywhn[:, 2:].cpu().numpy() 
            boxes = (boxes*factor).astype(np.uint16)
            scores = results.boxes.conf
            classes = results.boxes.cls

            # class names
            class_names = utils.read_class_names(f"./data/classes/coco.names")
            find_classes = ['car', 'truck']

            names = []
            indx = []

            for j in range(len(classes)):
                class_indx = int(classes[j])
                class_name = class_names[class_indx]
                if class_name in find_classes:
                    indx.append(j)
                    names.append('car')
            bboxes = boxes[indx]
            scores = scores[indx]

            # encode yolo detections and feed to tracker
            features = encoder_sort(frame[i], bboxes)
            detections = [Detection(bbox, score, class_name, feature)
                        for bbox, score, class_name, feature
                        in zip(bboxes, scores, names, features)]       
        
            # run nms
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[j] for j in indices] 
            
            # Call the tracker
            tracker[i].predict()
            tracker[i].update(detections)
        
        query_idx = -1
        query_size = 0
        query_bbox = None
        query_feat = None
        gallery_idxs = []
        gallery_feat = None
        gallery_bbox = []
        gallery_match_ids= []
        
        for i, query in enumerate(tracker[1].tracks):
            if not query.is_confirmed() or query.time_since_update > 1:
                continue
            x,y,w,h = query.to_tlwh()
            
            if w*h > query_size:
                query_idx = i
                query_size = w*h
                query_bbox = query.to_tlwh()[np.newaxis, ...]
            
        for i, gallery in enumerate(tracker[0].tracks):
            if not gallery.is_confirmed() or gallery.time_since_update > 1:
                continue
            gallery_idxs.append(i)
            gallery_bbox.append(gallery.to_tlwh())
            gallery_match_ids.append(gallery.match_id)
               
        if query_bbox is not None and gallery_bbox:
            gallery_bbox = np.stack(gallery_bbox)
            query_feat = encoder_reid(frame2, query_bbox)
            gallery_feat = encoder_reid(frame1, gallery_bbox)
        
            cosine_distance = nn_matching._cosine_distance(query_feat, gallery_feat)[0, :]
            idxsort = np.argsort(cosine_distance)
            idx_threshold = idxsort[np.where(cosine_distance[idxsort] < min_cosine_distance)]
            
            for idx in idx_threshold:
                gallery_idx = gallery_idxs[idx]
                dist = cosine_distance[idx]
                query_id = tracker[1].tracks[query_idx].match_id
                
                if tracker[1].tracks[query_idx].match_distance > dist and tracker[0].tracks[gallery_idx].match_distance > dist:
                    if tracker[1].tracks[query_idx].matched:
                        for i, _match_id in enumerate(gallery_match_ids):
                            if _match_id == query_id:
                                tracker[0].tracks[gallery_idxs[i]].matched = False
                                tracker[0].tracks[gallery_idxs[i]].match_id = -1
                                tracker[0].tracks[gallery_idxs[i]].match_distance = 1.0
                        tracker[0].tracks[gallery_idx].matched = True
                        tracker[0].tracks[gallery_idx].match_id = query_id
                        tracker[0].tracks[gallery_idx].match_distance = dist
                        tracker[1].tracks[query_idx].match_distance = dist
                    else:
                        match_id += 1
                        tracker[1].tracks[query_idx].matched = True
                        tracker[1].tracks[query_idx].match_id = match_id
                        tracker[1].tracks[query_idx].match_distance = dist
                        tracker[0].tracks[gallery_idx].matched = True
                        tracker[0].tracks[gallery_idx].match_id = match_id
                        tracker[0].tracks[gallery_idx].match_distance = dist
                    break
                
        # update tracks
        for track in tracker1.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1,y1,x2,y2 = track.to_tlbr().astype(np.int16)
            class_name = track.get_class()
        
            if track.matched:
                color = colors[int(track.match_id) % len(colors)]
                color = [i * 255 for i in color]
                string = f"[car : {str(track.match_id)}]"
                cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame1, (x1, y1-30), ((x1)+(len(string)*13), y1), color, -1)
                cv2.putText(frame1, string,(x1, y1-10),0, 0.75, (255,255,255),2) 

            elif FLAGS.plot_tracking:
                # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame1, (x1, y1-30), ((x1)+(len(class_name)+len(str(track.track_id)))*17, y1), color, -1)
                cv2.putText(frame1, class_name + "-" + str(track.track_id),(x1, y1-10),0, 0.75, (255,255,255),2)      
            max_instance[0] = max(max_instance[0], track.track_id)
    
        frame2_resized = cv2.resize(frame2, (int(v2width//4), int(v2height//4)))

        # update tracks
        for track in tracker2.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1,y1,x2,y2 = track.to_tlbr().astype(np.int16)
            class_name = track.get_class()
            
            if track.matched:
                color = colors[int(track.match_id) % len(colors)]
                color = [i * 255 for i in color]
                string = f"[car : {str(track.match_id)}]"
                cv2.rectangle(frame2_resized, (x1//4, y1//4), (x2//4, y2//4), color, 2)
                cv2.rectangle(frame2_resized, (x1//4, y1//4-30), ((x1//4)+(len(string))*13, y1//4), color, -1)
                cv2.putText(frame2_resized, string,(x1//4, y1//4-10),0, 0.75, (255,255,255),2)                          
            elif FLAGS.plot_tracking:
        # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame2_resized, (x1//4, y1//4), (x2//4, y2//4), color, 2)
                cv2.rectangle(frame2_resized, (x1//4, y1//4-30), ((x1//4)+(len(class_name)+len(str(track.track_id)))*17, y1//4), color, -1)
                cv2.putText(frame2_resized, class_name + "-" + str(track.track_id),(x1//4, y1//4-10),0, 0.75, (255,255,255),2)      
            max_instance[1] = max(max_instance[1], track.track_id)
        
        frame_total = frame1.copy()
        frame_total[:int(height[1]//4), int(width[0]-width[1]//4):, :] = frame2_resized

        # calculate frames per second of running detections
        result = np.asarray(frame_total)
        result = cv2.cvtColor(frame_total, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()       
            
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
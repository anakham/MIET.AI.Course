import argparse
import cv2 as cv
import numpy as np
from pathlib import Path
import os


def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError
    
def visualize(input, faces, face_ids, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            face_label = face_ids[idx]
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.putText(input, face_label, (coords[0], coords[1] + coords[3]+24), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 24), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)


def get_embedings(img : np.array, detector : cv.FaceDetectorYN, recognizer : cv.FaceRecognizerSF, scale, face_pose = None) ->np.array:
    face = face_pose
    global face_cnt
    if face_pose is None:
        imgWidth = int(img.shape[1] * scale)
        imgHeight = int(img.shape[0] * scale)
        img = cv.resize(img, (imgWidth, imgHeight))
        detector.setInputSize((imgWidth, imgHeight))
        faces = detector.detect(img)
        if faces is None:
            return None
        face = faces[1][0]
    if face is None:
        return None
    face_align = recognizer.alignCrop(img, face)
    #cv.imshow("debug", face_align)
    face_feature = recognizer.feature(face_align)
    return face_feature


def get_embedings_from_file(face_file : str, detector : cv.FaceDetectorYN, recognizer : cv.FaceRecognizerSF, scale) ->np.array:
    try:
        img = cv.imread(face_file)
        return get_embedings(img, detector, recognizer, scale)
    except:    
       return None


def find_match_score(face1_feature, face2_feature, recognizer):
        
    cosine_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_COSINE)
    l2_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_NORM_L2)

    return (cosine_score, l2_score)


def make_desicion(cosine_score, l2_score):
    cosine_similarity_threshold = 0.363
    l2_similarity_threshold = 1.128
    msg = 'different identities'
    match_result = False
    if cosine_score >= cosine_similarity_threshold:
        msg = 'the same identity'
        match_result = True
    print('They have {}. Cosine Similarity: {}, threshold: {} (higher value means higher similarity, max 1.0).'.format(msg, cosine_score, cosine_similarity_threshold))
    msg = 'different identities'
    if l2_score <= l2_similarity_threshold:
        msg = 'the same identity'
        match_result = True
    print('They have {}. NormL2 Distance: {}, threshold: {} (lower value means higher similarity, min 0.0).'.format(msg, l2_score, l2_similarity_threshold))
    return match_result



def find_best_match(face_db : dict, face_feature : np.array, recongnizer : cv.FaceRecognizerSF):
    max_cos_score = 0
    max_l2_score = 1e20
    arg_max = ""
    for label, feature in face_db.items():
        (cos_score, l2_score) = find_match_score(face_feature, feature, recognizer)
        if cos_score > max_cos_score:
            arg_max = label
            max_cos_score = cos_score
            max_l2_score = l2_score
    if make_desicion(max_cos_score, max_l2_score):
        return arg_max
    else:
        return ""
    

def scan_folder(face_folder : str, detector : cv.FaceDetectorYN, recognizer: cv.FaceRecognizerSF, scale) -> dict:
    faces = dict()
    face_dir_path = Path(face_folder)
    for face_file in os.listdir(face_folder):
        face_features = get_embedings_from_file( str((face_dir_path / face_file).resolve()) , detector, recognizer, scale)
        if face_features is None:
            continue
        file_name = Path(face_file).stem
        faces[file_name] = face_features

    return faces


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', type=str, default='face_video.mkv', help='Path to the input video.')
    parser.add_argument('--face_folder', '-fc', default='faces', type=str, help='Path to the folder with face dataset.')
    parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
    parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2022mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
    parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
    parser.add_argument('--score_threshold', type=float, default=0.95, help='Filtering out faces of score < score_threshold.')
    parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
    parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')

    args = parser.parse_args()

    cap = cv.VideoCapture(args.video)
    #cap = cv.VideoCapture(0)
    tm = cv.TickMeter()
    
    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    recognizer = cv.FaceRecognizerSF.create(args.face_recognition_model,"")
    
    face_db = scan_folder(args.face_folder, detector, recognizer, args.scale)
    

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)*args.scale)
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)*args.scale)
    detector.setInputSize([frameWidth, frameHeight])
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break
        frame = cv.resize(frame, (frameWidth, frameHeight))
        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        face_ids = []
        if faces[1] is not None:
            for face in faces[1]:
                face_feature = get_embedings(frame, detector, recognizer, args.scale, face)
                face_ids.append(find_best_match(face_db, face_feature, recognizer))
        tm.stop()
        # Draw results on the input image
        visualize(frame, faces, face_ids, tm.getFPS())
        # Visualize results
        cv.imshow('Live', frame)
    
    cv.destroyAllWindows()
<<<<<<< HEAD
import os
import argparse
import cv2
import mediapipe as mp

def proceess_img(img, facedetection):

    H, W, _ = img.shape

    img_rgp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgp)



    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            #img = cv2.rectangle(img, (x1,y1), (x1 + w, y1 + h), (0,255,0),10) # for face detection

            # blur_face

            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))
    return img

    """cv2.imshow('img',img)
    cv2.waitKey(0)"""


args = argparse.ArgumentParser()

args.add_argument("--mode",default='webcam')
args.add_argument("--filepath",default=None)

args = args.parse_args()

output_dir = '/output'
if not os.path.exists((output_dir)):
    os.makedirs(output_dir)


# detect_faces
np_face_detection = mp.solutions.face_detection

with np_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5 ) as face_detection:

    if args.mode in ["image"]:
        # read image
        img = cv2.imread(args.filepath)


        img = proceess_img(img,face_detection)


        #save image
        cv2.imwrite(os.path.join(output_dir,"output.jpg"),img)

    elif args.mode in ['video']:



        cap = cv2.VideoCapture(args.filepath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_dir, "output.mp4"),
                                       cv2.Videowriter_fourcc(*'MP4'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))

        while ret:
            frame = proceess_img(frame, face_detection)

            output_video.write(frame)

            ret, frame = cap.read()


        cap.release()
        output_video.release()

    elif args.mode in ["webcam"]:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        while ret:
            frame = proceess_img(frame,face_detection)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()

=======
import os
import argparse
import cv2
import mediapipe as mp

def proceess_img(img, facedetection):

    H, W, _ = img.shape

    img_rgp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgp)



    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            #img = cv2.rectangle(img, (x1,y1), (x1 + w, y1 + h), (0,255,0),10) # for face detection

            # blur_face

            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))
    return img

    """cv2.imshow('img',img)
    cv2.waitKey(0)"""


args = argparse.ArgumentParser()

args.add_argument("--mode",default='webcam')
args.add_argument("--filepath",default=None)

args = args.parse_args()

output_dir = '/output'
if not os.path.exists((output_dir)):
    os.makedirs(output_dir)


# detect_faces
np_face_detection = mp.solutions.face_detection

with np_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5 ) as face_detection:

    if args.mode in ["image"]:
        # read image
        img = cv2.imread(args.filepath)


        img = proceess_img(img,face_detection)


        #save image
        cv2.imwrite(os.path.join(output_dir,"output.jpg"),img)

    elif args.mode in ['video']:



        cap = cv2.VideoCapture(args.filepath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_dir, "output.mp4"),
                                       cv2.Videowriter_fourcc(*'MP4'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))

        while ret:
            frame = proceess_img(frame, face_detection)

            output_video.write(frame)

            ret, frame = cap.read()


        cap.release()
        output_video.release()

    elif args.mode in ["webcam"]:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        while ret:
            frame = proceess_img(frame,face_detection)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()

>>>>>>> 0e4683d6fdcbd607ec7c8461bf9a89bd95c66141

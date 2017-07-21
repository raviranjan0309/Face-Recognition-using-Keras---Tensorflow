import os
import cv2
import dlib
import scipy.misc

face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

workDirectory = os.path.dirname(__file__)

trainDirectory = os.path.join(workDirectory,'images')

celebrityTrainFolder = [os.path.join(trainDirectory, f) for f in os.listdir(trainDirectory)]

count = 0

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def get_face_detect(imagePath):
    print(count, imagePath)
    image = scipy.misc.imread(imagePath)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    for face_pose in shapes_faces:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = rect_to_bb(face_pose.rect)
        image = cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        crop_img = image[y:y + h, x:x + w]
        #if (v != 0 for v in (x,y,w,h)):
        crop_img = cv2.resize(crop_img, (200, 200), interpolation=cv2.INTER_CUBIC)
        return crop_img


for i in celebrityTrainFolder:
    imagePath = [os.path.join(i,f) for f in os.listdir(i)]
    for f in imagePath:
        sub_face = get_face_detect(f)
        celebrity = None
        if 'deepika' in f:
            celebrity = 'deepika'
        else:
            celebrity = 'ranveer'
        face_file_name = workDirectory + "/train/" + celebrity + "/" + celebrity+ "_" + str(count) + ".jpg"
        count = count + 1
        cv2.imwrite(face_file_name, sub_face)
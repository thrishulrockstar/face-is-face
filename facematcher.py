import dlib
import cv2


face_detector = dlib.get_frontal_face_detector()
face_recognition_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")


image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")


faces1 = face_detector(image1, 1)
faces2 = face_detector(image2, 1)


if len(faces1) != 1 or len(faces2) != 1:
  print("Error: both images must contain exactly one face.")
  exit()


face1 = face_recognition_model.compute_face_descriptor(image1, faces1[0])
face2 = face_recognition_model.compute_face_descriptor(image2, faces2[0])

distance = dlib.vector_distance(face1, face2)


if distance < 0.6:
  print("The two images contain the same person.")
else:
  print("The two images do not contain the same person.")

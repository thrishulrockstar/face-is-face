# face-is-face
face detector script which use dlib's to detect the faces in the two images, and then use dlib's face recognition model to extract a 128-dimensional face descriptor for each face.
The distance between the two face descriptors is then calculated using the Euclidean distance. If the distance is less than 0.6, the images are considered a match.

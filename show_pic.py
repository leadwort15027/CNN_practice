###############################
#  python show_pic.py NUMBER  #
###############################
import cv2
from keras.datasets import cifar10
import sys
output = {
0:"airplane",
1:"automobile",
2:"bird",
3:"cat",
4:"deer",
5:"dog",
6:"forg",
7:"horse",
8:"ship",
9:"truck"
}

(X,Y),(x,y) = cifar10.load_data()
img = cv2.resize(X[int(sys.argv[1])],(256,256))
cv2.imshow(output.get(int(Y[int(sys.argv[1])])),img)
cv2.waitKey(0)
cv2.destroyAllWindows()

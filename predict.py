#############################################################
#  python predict.py  MODEL_FILE_NAME.h5 PREDICT_IMAGE.jpg  #
#############################################################
import cv2
import h5py
import numpy as np
import sys
from keras.models import load_model
from keras.models import Sequential

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

max = 0
my_model = load_model(sys.argv[1])
img = cv2.imread(sys.argv[2])
cv2.imshow("origin",img)
img = cv2.resize(img,(32,32))
img = img.astype('float32')
img /= 255
img_ = img
img = np.expand_dims(img, axis=0)
predictions = my_model.predict(img)
for x in predictions:
	max = np.argmax(x)
	print "-------------------------------------------"
	for i in range(len(x)):
		#print(output.get(i)+" : %f")%x[i]
		print("%-10s : %f")%(output.get(i),x[i])
print "-------------------------------------------"

cv2.imshow(output.get(max),cv2.resize(img_,(200,200)))
cv2.waitKey(0)
cv2.destroyAllWindows()

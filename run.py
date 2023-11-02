import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import os, cv2


mainPath = os.getcwd() + "\\Happy Sad Classifier\\happy_sad-Classifier\\"
modelPath = mainPath + "HappySadModel.h5"
model = load_model(modelPath)
file = os.listdir(mainPath + "Image\\")

a = 1
img = cv2.imread(mainPath + "\\Image\\" + file[a])
imgAlt = cv2.resize(img, (32,32),interpolation = cv2.INTER_AREA) / 255
x = np.expand_dims(imgAlt, axis = 0)



pred = model.predict(x)

print("File: " + file[a] + "\n")
plt.imshow(img[:,:,::-1]) #[:,:,::-1] for image to be in rgb
if(pred < 0.5):
    plt.title("Predicted: Sad")
else:
    plt.title("Predicted: happy")
plt.axis("off")
plt.show()

# for prob in range(len(pred[0])):
#     print(labels[prob] + ": " + str(round(pred[0][prob] * 100 , 3)))
# print("************************\n")
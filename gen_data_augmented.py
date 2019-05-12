from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["monkey", "crow", "boar"]

num_classes = len(classes)
image_size = 50
num_testdata= 100

X_train = []
X_test = []
y_train = []
y_test = []

for index, classlabel in enumerate(classes):
  photos_dir = "./" + classlabel
  files = glob.glob(photos_dir + "/*.jpg")
  for i, file in enumerate(files):
    if i >= 200: break
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)

    if i < num_testdata:
      X_test.append(data)
      y_test.append(index)
    else:
      for angle in range(-20, 20, 5):
        img_r = image.rotate(angle)
        data = np.asarray(img_r)
        X_train.append(data)
        y_train.append(index)

        img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
        data = np.asarray(img_trans)
        X_train.append(data)
        y_train.append(index)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./animal_aug.npy", xy)
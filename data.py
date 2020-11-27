import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv("sign_mnist_train_custom.csv")

count = 0
shape_before = df_train.shape

import os

for dirname, _, filenames in os.walk('./data/D/'):
    for filename in filenames:
        count += 1
        path = os.path.join(dirname, filename)
        print(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(28*28)
        img = np.insert(img, 0, 3, axis=0)
        df_train = df_train.append(pd.DataFrame([img],columns=df_train.columns), ignore_index=True )

for dirname, _, filenames in os.walk('./data/E/'):
    for filename in filenames:
        count += 1
        path = os.path.join(dirname, filename)
        print(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(28*28)
        img = np.insert(img, 0, 4, axis=0)
        df_train = df_train.append(pd.DataFrame([img],columns=df_train.columns), ignore_index=True )

for dirname, _, filenames in os.walk('./data/F/'):
    for filename in filenames:
        count += 1
        path = os.path.join(dirname, filename)
        print(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(28*28)
        img = np.insert(img, 0, 5, axis=0)
        df_train = df_train.append(pd.DataFrame([img],columns=df_train.columns), ignore_index=True )

print(shape_before)
print(count)
print(df_train.shape)
df_train.to_csv("sign_mnist_train_custom2.csv",index=False)

# path1 = "./data/A/A1.jpg"
# # path2 = "./data/B/B1.jpg"
# # path3 = "./data/C/C1.jpg"

# img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
# # img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
# # img3 = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)

# # print(df_train.shape)
# # print(df_train.iloc[-1,:])
# # img1_tmp = img1.reshape(28*28)
# # img1_tmp = np.insert(img1_tmp, 0, 0, axis=0)
# # df_train = df_train.append(pd.DataFrame([img1_tmp],columns=df_train.columns), ignore_index=True )
# # print(df_train.shape)
# # print(df_train.iloc[-1,:].values)

# fig,axe=plt.subplots(2,2)
# fig.suptitle('Preview of dataset')
# # axe[0][0].imshow(df_train.drop(['label'],axis=1).iloc[-1,:].values.reshape(28,28),cmap='gray')
# axe[0][1].imshow(img1,cmap='gray')
# # axe[1][0].imshow(img3,cmap='gray')
# plt.show()

# import time
# time.sleep(10)
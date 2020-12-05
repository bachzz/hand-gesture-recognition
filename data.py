import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv("sign_mnist_train_custom.csv")

count = 0
shape_before = df_train.shape

import os

chars = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# cur: 17,R -> 24,Y

for i in range(0,25):
    if i != 9:
        for dirname, _, filenames in os.walk('./data/'+chars[i]+'/'):
            for filename in filenames:
                count += 1
                path = os.path.join(dirname, filename)
                print(path)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = img.reshape(28*28)
                img = np.insert(img, 0, i, axis=0)
                df_train = df_train.append(pd.DataFrame([img],columns=df_train.columns), ignore_index=True )

# for dirname, _, filenames in os.walk('./data/E/'):
#     for filename in filenames:
#         count += 1
#         path = os.path.join(dirname, filename)
#         print(path)
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img = img.reshape(28*28)
#         img = np.insert(img, 0, 4, axis=0)
#         df_train = df_train.append(pd.DataFrame([img],columns=df_train.columns), ignore_index=True )

# for dirname, _, filenames in os.walk('./data/F/'):
#     for filename in filenames:
#         count += 1
#         path = os.path.join(dirname, filename)
#         print(path)
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img = img.reshape(28*28)
#         img = np.insert(img, 0, 5, axis=0)
#         df_train = df_train.append(pd.DataFrame([img],columns=df_train.columns), ignore_index=True )

print(shape_before)
print(count)
print(df_train.shape)
df_train.to_csv("sign_mnist_train_custom2.csv",index=False)

# fig,axe=plt.subplots(2,2)
# fig.suptitle('Preview of dataset')
# # axe[0][0].imshow(df_train.drop(['label'],axis=1).iloc[-1,:].values.reshape(28,28),cmap='gray')
# axe[0][1].imshow(img1,cmap='gray')
# # axe[1][0].imshow(img3,cmap='gray')
# plt.show()

# import time
# time.sleep(10)
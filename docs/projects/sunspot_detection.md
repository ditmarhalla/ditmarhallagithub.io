# Detecting sunspots from a picture of the sun

This is a fun project I started on a weekend. The idea is simple: Get an image and mark the sunspos with a green square.

For this task I noticed online that *cv2* was the best option to work with picture files.

In the begining we load the file using the integreted function of *cv2*. After that we visualize it.
Then the fun part starts. using the function in line 20  we threshold the image so that only the sunspots remain.

After the threshholding we have to convert the data into something we can work with. *cv2* has a fucntion for this that converts the data into NumPy array.
You can see that because we only need the sunspots we have to use the circle equation (line 36) to make sure that we do not detect the sun itself.
The next integreted function (line 43) gets us the surface of each sunspot and then we use a *for* loop to mark each sunspot with a green rectangle (line 33).

From the pictures below you can see that from picture 1 that is the input we get the output with he detection.

![Screenshot](https://github.com/ditmarhalla/astronomy/blob/main/sunspot_detection/sunspot1.jpg?raw=true)

![Screenshot](https://github.com/ditmarhalla/astronomy/blob/main/sunspot_detection/Finish.png?raw=true)


```python
import os
import cv2 # opencv library
import numpy as np
import matplotlib.pyplot as plt

"""Make the pwd implementation"""
cwd = os.getcwd()
file = "\sunspot1.jpg"
path = cwd + file
image = cv2.imread(path,0)

image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# perform image thresholding
ret, thresh = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)

# find contours
contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
valid_cntrs = []

for i,cntr in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cntr)
    if ((x-249)**2 + (y-249)**2)<= 238**2:
        valid_cntrs.append(cntr)
"""implement image size detection for the contur LINE 36"""

# count the number of dicovered sunspots
print("The number of sunspots is: ",len(valid_cntrs))

contour_sizes = [(cv2.contourArea(contour), contour) for contour in valid_cntrs]

for i in range(len(valid_cntrs)):
    x,y,w,h = cv2.boundingRect(contour_sizes[i][1])
    final = cv2.rectangle(image_1,(x,y),(x+w,y+h),(0,255,0),1)

plt.imshow(final)
plt.show()
```

This code is not perfect and has some problems but its a nice first implementation for a weekend project for a newbie python programer.

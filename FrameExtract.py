import cv2
import tkinter as tk
import os
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
print(file_path)
tmp=file_path.split('/')[-1].split('.')[0]
print(tmp)
if not os.path.exists(tmp):
    os.makedirs(tmp)
vidcap = cv2.VideoCapture(file_path)
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  if(image is None or len(image)==0):
    break
  print('Read a new frame: ', success)
  cv2.imwrite(tmp+"/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1
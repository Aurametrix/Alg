#!/usr/bin/python3

#coding: utf-8

import os
import shutil
import glob
import math
import argparse
import warnings
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
        
class K_means:

  def __init__(self, k=3):
    self.k = k
    self.cluster = []
    self.data = []
    self.end = []
    self.i = 0
    self.size = False

  def manhattan_distance(self,x1,x2):
    s = 0.0
    for i in range(len(x1)):
      s += abs( float(x1[i]) - float(x2[i]) )
    return s

  def euclidian_distance(self,x1,x2):
    s = 0.0
    for i in range(len(x1)):
      s += math.sqrt((float(x1[i]) - float(x2[i])) ** 2)
    return s

  def read_image(self,im):
    if self.i >= self.k :
      self.i = 0
    try:
      img = mpimg.imread(im)
      v = [float(p)/float(img.size)*100 for p in np.histogram(img)[0]]
      if self.size :
        v += [img.shape[0], img.shape[1]]
      self.i += 1
      pbar.update(1)
      return [self.i, v, im]
    except:
      print("Error reading ",im)
      return [None,None,None]

  def generate_k_means(self):
    final_mean = []
    for c in range(self.k):
      partial_mean = []
      for i in range(len(self.data[0])):
        s = 0.0
        t = 0
        for j in range(len(self.data)):
          if self.cluster[j] == c :
            s += self.data[j][i]
            t += 1
        if t != 0 :
          partial_mean.append(float(s)/float(t))
        else:
          partial_mean.append(float('inf'))
      final_mean.append(partial_mean)
    return final_mean

  def generate_k_clusters(self,folder,size):
    pool = ThreadPool(cpu_count())
    self.size = size
    result = pool.map(self.read_image, folder)
    self.cluster = [r[0] for r in result if r[0]]
    self.data = [r[1] for r in result if r[1]]
    self.end = [r[2] for r in result if r[2]]

  def rearrange_clusters(self):
    acabou = False
    while(acabou == False):
      acabou = True
      m = self.generate_k_means()
      for x in range(len(self.cluster)):
        dist = []
        for a in range(self.k):
          dist.append( self.manhattan_distance(self.data[x],m[a]) )
        if self.cluster[x] != dist.index(min(dist)) :
          self.cluster[x] = dist.index(min(dist))
          acabou = False

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="path to input dataset")
ap.add_argument("-k", "--kmeans", type=int, default=5, help="how many groups")
ap.add_argument("-s", "--size", default=False, action="store_true", help="use size to compare images")
ap.add_argument("-m", "--move", default=False, action="store_true", help="move instead of copy")
args = vars(ap.parse_args())

types = ('*.jpg', '*.JPG', '*.png', '*.jpeg')
imagePaths = []
for files in types:
  imagePaths.extend(sorted(glob.glob(args["folder"]+files)))

if(len(imagePaths) <= 0):
  print("Nenhuma imagem encontrada!")
  exit()

pbar = tqdm(total=len(imagePaths))

k = K_means(args["kmeans"])

k.generate_k_clusters(imagePaths,args["size"])

k.rearrange_clusters()

folder = args["folder"]

for i in range(k.k):
  os.makedirs(folder+""+'{:04}'.format(i+1))

cmd = 'cp'
if args["move"] :
  cmd = 'mv'

for i in range(len(k.cluster)):
  src = k.end[i]
  dst = folder+"/"+'{:04}'.format(k.cluster[i]+1)+"/"
  if args["move"] :
    shutil.move(src, dst)
  else :
    shutil.copy(src, dst)


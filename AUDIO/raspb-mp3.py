import os
import glob
import subprocess
from time import sleep
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(22,GPIO.IN)   # pin 15
GPIO.setup(23,GPIO.IN)   # pin 16
GPIO.setup(24,GPIO.IN)   # pin 18
GPIO.setup(25,GPIO.IN)   # pin 22

sleep(20)
os.chdir('/media/HANIEH')
f = glob.glob('*.mp3')
#print f
h = len(f)
flag=1
pt=0
st=0

while True:
    if flag==1:
        player = subprocess.Popen(["omxplayer",f[pt]],stdin=subprocess.PIPE) #,stdout=subprocess.PIPE,stderr=subprocess.PIPE
        fi = player.poll()
        flag=0
        st=0

    if (GPIO.input(22)==False):
        sleep(0.5)
        fi = player.poll()
        if fi!=0:
            player.stdin.write("p")      # pin 15 pause



    if (GPIO.input(23)==False):
        sleep(0.5)
        fi = player.poll()
        if fi!=0:
            player.stdin.write("q")      # pin 16 stop
            st=1

    if (GPIO.input(24)==False):
        if st==0:
            player.stdin.write("q")      # pin 18 Next Audio
        flag=1
        pt = pt+1
        if pt>h-1:
            pt=0
        sleep(0.5)    

    elif (GPIO.input(25)==False):
        if st==0:
            player.stdin.write("q")      # pin 22 Next Audio
        flag=1
        pt = pt-1
        if pt<0:
            pt=h-1
        sleep(0.5)

    else:
        fi = player.poll()
        if (fi==0 and st==0):
            flag=1
            pt=pt+1
            if pt>h-1:
                pt=0
    sleep(0.1)


#player.stdin.write("+")

#player.stdin.write("-")﻿

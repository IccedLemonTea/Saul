#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#   
# This is where the main adjustments occured to create SAUL during the summer of 2023

from jetson_utils import cudaAllocMapped, cudaToNumpy
from pprint import pprint
import numpy as np
import time
import serial
turn_counter=0
turning=False
navigating=True
exit_row=False
enter_row=False
turnclock=0
arduino = serial.Serial(
port = '/dev/ttyUSB0',
baudrate=115200,
bytesize=serial.EIGHTBITS,
parity = serial.PARITY_NONE,
stopbits = serial.STOPBITS_ONE, 
timeout = 0,
xonxoff = False,
rtscts = False,
dsrdtr= False,
write_timeout=0
)

class segmentationBuffers:

    def __init__(self, net, args):
        self.net = net
        self.mask = None
        self.overlay = None
        self.composite = None
        self.class_mask = None
        
        self.use_nav = args.nav
        self.use_mask = "mask" in args.visualize
        self.use_overlay = "overlay" in args.visualize
        self.use_composite = self.use_mask and self.use_overlay
        
        if not self.use_overlay and not self.use_mask:
            raise Exception("invalid visualize flags - valid values are 'overlay' 'mask' 'overlay,mask'")
             
        self.grid_width, self.grid_height = net.GetGridSize()	
        self.num_classes = net.GetNumClasses()

    @property
    def output(self):
        if self.use_overlay and self.use_mask:
            return self.composite
        elif self.use_overlay:
            return self.overlay
        elif self.use_mask:
            return self.mask
            
    def Alloc(self, shape, format):
        if self.overlay is not None and self.overlay.height == shape[0] and self.overlay.width == shape[1]:
            return

        if self.use_overlay:
            self.overlay = cudaAllocMapped(width=shape[1], height=shape[0], format=format)

        if self.use_mask:
            mask_downsample = 2 if self.use_overlay else 1
            self.mask = cudaAllocMapped(width=shape[1]/mask_downsample, height=shape[0]/mask_downsample, format=format) 

        if self.use_composite:
            self.composite = cudaAllocMapped(width=self.overlay.width+self.mask.width, height=self.overlay.height, format=format) 

        if self.use_nav:
            self.class_mask = cudaAllocMapped(width=self.grid_width, height=self.grid_height, format="gray8")
            self.class_mask_np = cudaToNumpy(self.class_mask)

    def navigation(self):
        #turn_counter=0
        if not self.use_nav:
            return
        global navigating
        # get the class mask (each pixel contains the classID for that grid cell)
        self.net.Mask(self.class_mask, self.grid_width, self.grid_height)
        mask_array=cudaToNumpy(self.class_mask)
        """for row in mask_array:
            print(*row, sep="\t")
            #print(" ")"""
        postCheck=mask_array[:,1:8]
        col1=mask_array[:,1]
        col2=mask_array[:,2]
        col3=mask_array[:,3]
        col4=mask_array[:,4]
        col5=mask_array[:,5]
        col6=mask_array[:,6]
        col7=mask_array[:,7]
        col8=mask_array[:,8]
        path1=mask_array[5,3]
        path2=mask_array[5,4]
        path3=mask_array[5,5]
        path4=mask_array[6,3]
        path5=mask_array[6,4]
        path6=mask_array[6,5]
        postL=mask_array[:,0]
        postR=mask_array[:,9]
        botrow=mask_array[9,:]
        if navigating:
            global turnclock
            turnclock+=1
            print("navigating")
            if path1 == 1 and path1 == 1 and path3 == 1 and path4 == 1 and path5 == 1 and path6 == 1:
                arduino.write("1".encode())
                data=arduino.readline()
                #print("continue")

            elif any(col1==1) and path2!=1 and path3!=1 and path5!=1 and path6!=1:
                arduino.write("2".encode())
                data=arduino.readline()
                #print("Turn Left")

            elif any(col8==1) and path1!=1 and path2!=1 and path4!=1 and path5!=1:
                arduino.write("3".encode())
                data=arduino.readline()
                #print("Turn Right")
                #elif any(last_seenR!=3) and any(last_seenL!=3):

            if any(col8==3) and any(col1==3) and any(postL==3) and any(postR==3):
                navigation = True

            else:
                if (turnclock>=400):
                    if any(postL==2) or any(postR==2):
                        global turn_counter
                        navigating=False
                        global exit_row
                        exit_row=True
                        arduino.write("4".encode())
                        turnclock=0

        if exit_row:
            print("driving slow")
            if all(botrow==1):
                print("seen end of row")
                global turning
                turning=True
                exit_row=False

        if turning:
                if (turn_counter%2)==0:
                    arduino.write("2".encode())
                    print("turning left")
                    if any(postL==2):
                        print("seen post on left")
                        arduino.write("1".encode())
                        time.sleep(0.5)
                        global enter_row
                        enter_row=True
                        exit_row=False
                        navigating=False
                        turning=False
                else:
                    arduino.write("3".encode())
                    if any(postR==2):
                        print("seen post on right")
                        arduino.write("1".encode())
                        time.sleep(0.5)
                        enter_row=True
                        exit_row=False
                        navigating=False
                        turning=False

        if enter_row:
            print("entering row")
            if (turn_counter%2)==0:
                arduino.write("2".encode())
                if any(postR==2):
                    arduino.write("0".encode())
                    exit_row=False
                    enter_row=False
                    navigating=True
                    turn_counter+=1
            else:
                arduino.write("3".encode())
                if any(postL==2):
                    arduino.write("0".encode())
                    exit_row=False
                    enter_row=False
                    navigating=True
                    turn_counter+=1


 



        """if self.class_mask[:,1]==2 and pixel2==2 and pixel3==2 and pixel4==2:
            print("continue")
        elif pixel1==2 and pixel3==2 and pixel2!=2 and pixel4!=2:
            print("turn left")
        elif pixel1!=2 and pixel3!=2 and pixel2==2 and pixel4==2:
            print("turn right")
        else:
            print("idk")
        #print(mask_array)
        time.sleep(1)"""


        """for y in range(self.grid_width):
            for x in range(self.grid_width):
                pixel = self.class_mask[5,5]
                print(pixel)"""


        #compute the number of times each class occurs in the mask
        #class_histogram, _ = np.histogram(self.class_mask_np, bins=self.num_classes, range=(0, self.num_classes-1))

        #print('grid size:   {:d}x{:d}'.format(self.grid_width, self.grid_height))
        #print('num classes: {:d}'.format(self.num_classes))
      
        #print('-----------------------------------------')
        #print(' ID  class name        count     %')
        #print('-----------------------------------------')

        #for n in range(self.num_classes):
            #percentage = float(class_histogram[n]) / float(self.grid_width * self.grid_height)
            #print(' {:>2d}  {:<18s} {:>3d}   {:f}'.format(n, self.net.GetClassDesc(n), class_histogram[n], percentage)) 


# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:35:50 2021

@author: ADARSH PATHAK
"""


from tkinter import *
from PIL import Image,ImageTk
import tkinter.font as font
import os

root=Tk()
root.title("Hand Gesture Recognition")
root.geometry('750x432')
root.resizable(height = None, width = None)
labelFont=font.Font(size=20)

load = ImageTk.PhotoImage(Image.open("C:\\Users\\ADARSH PATHAK\\Desktop\\PW - II\\number cnn\\Deaf-students-cover1.jpg"))
render=load
img=Label(root,image= render)
img.place(x=0,y=0)

#label= Label(root,text="Hand Gestures Recognition",width=100,height=10)
#label.place(x=0,y=0)
#label['font']=labelFont

def run1():
    print("Predict by CNN called")
    os.system('python num_predict.py')
    
def run2():
    print("Predict by Contour called")
    os.system('python Contour.py')


B1= Button(root, text ="Predict by CNN",bg="black",fg="white",width=30,height=3,command = run1)
B1.place(relx=0,x=200,y=380,anchor=CENTER)


B2= Button(root, text ="Predict by Contour",bg="black",fg="white",width=30,height=3,command = run2)
B2.place(relx=0,x=520,y=380,anchor=CENTER)


load2 = ImageTk.PhotoImage(Image.open("C:\\Users\\ADARSH PATHAK\\Desktop\\PW - II\\number cnn\\Text.jpg"))
l1=Label(root,image=load2)
l1.place(x=25,y=30)

#label= Label(root,text="Hand Gestures Recognition",width=23,height=1)
#label.place(x=200,y=40)
#label['font']=labelFont


root.mainloop()

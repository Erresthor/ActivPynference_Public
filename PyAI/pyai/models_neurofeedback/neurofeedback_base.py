# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 2021

@author: cjsan
"""
import numpy as np
import random
from PIL import Image, ImageDraw
import sys,os,inspect
import math

from ..base.miscellaneous_toolbox import flexible_copy 
from ..base.function_toolbox import normalize
from ..layer.mdp_layer import mdp_layer

def centersize_to_box(center_position,size):
    top_x = int(center_position[0] - size[0]/2.0)
    top_y = int(center_position[1] - size[1]/2.0)
    bot_x = int(center_position[0] + size[0]/2.0)
    bot_y = int(center_position[1] + size[1]/2.0)
    return [(top_x,top_y),(bot_x,bot_y)]

class screen:
    def __init__(self,base_color,max_lev):
        self.level = 1
        self.max_level = max_lev
        self.col = base_color

    def update_level(self,new_level):
        self.level = new_level

    def draw(self,image_draw,center_position,size):
        # Base color when empty
        shap = centersize_to_box(center_position,size)
        image_draw.rectangle(shap,fill="grey")


        # The level jauge &  The level ticks: 
        # shap[0] changes depending on the level of the gauge
        offset = 0.1*size[1]
        counter = center_position[1] - size[1]/2.0 + offset


        # if(self.level == self.max_level) :
        #     image_draw.rectangle(shap,fill=self.col)

        if (self.max_level > 1):
            step = (size[1] - 2*offset)/(self.max_level - 1)
            for k in range(self.max_level,0,-1):
                if (self.level == k ):
                    jauge_shape = [(center_position[0] - size[0]/2.0,counter),(center_position[0] + size[0]/2.0,center_position[1]+size[1]/2.0)]
                    image_draw.rectangle(jauge_shape,fill=self.col)
                shape = [(center_position[0] - size[0]/2.0,counter),(center_position[0],counter)]
                
                image_draw.line(shape,fill="black")
                counter += step
        else :
            #Only one level, no need to display it
            image_draw.rectangle(shap,fill=self.col)

        shape = [(center_position[0] - size[0]/2.0,counter),(center_position[0],counter)]
        #image_draw.line(shape,fill="black")

        # The boundaries
        image_draw.rectangle(shap,fill=None,outline="black")

class multicolor_screen:
    def __init__(self,base_color,max_lev):
        self.level = np.ones((max_lev,))/max_lev
        self.max_level = max_lev
        self.col_0 = base_color

    def update_level(self,new_dist):
        self.level = new_dist

    def draw(self,image_draw,center_position,size):
        # Base color when empty
        shap = centersize_to_box(center_position,size)
        image_draw.rectangle(shap,fill="grey")


        # The level jauge &  The level ticks: 
        # shap[0] changes depending on the level of the gauge
        for level in range(self.max_level):
            print(level)
        offset = 0.1*size[1]
        counter = center_position[1] - size[1]/2.0 + offset


        # if(self.level == self.max_level) :
        #     image_draw.rectangle(shap,fill=self.col)

        if (self.max_level > 1):
            step = (size[1] - 2*offset)/(self.max_level - 1)
            for k in range(self.level.shape[0]):
                if (self.level == k ):
                    jauge_shape = [(center_position[0] - size[0]/2.0,counter),(center_position[0] + size[0]/2.0,center_position[1]+size[1]/2.0)]
                    image_draw.rectangle(jauge_shape,fill=self.col)
                shape = [(center_position[0] - size[0]/2.0,counter),(center_position[0],counter)]
                
                image_draw.line(shape,fill="black")
                counter += step
        else :
            #Only one level, no need to display it
            image_draw.rectangle(shap,fill=self.col)

        shape = [(center_position[0] - size[0]/2.0,counter),(center_position[0],counter)]
        #image_draw.line(shape,fill="black")

        # The boundaries
        image_draw.rectangle(shap,fill=None,outline="black")

class NF_model_displayer :
    def __init__(self,_ticks_mental,_ticks_screen):
        # Display elements :
        self.screen = screen("red",_ticks_screen)
        self.mental = screen("green",_ticks_mental)
        self.subjective = multicolor_screen("blue",_ticks_mental) 


    def update_values(self,screen_value,mental_value,new_dist=0):
        self.screen.update_level(screen_value)
        self.mental.update_level(mental_value)
        self.subjective.update_level(new_dist)

    def draw(self,size):
        x_size,y_size = size

        img = Image.new("RGB",(x_size,y_size),"white")
        draw = ImageDraw.Draw(img,"RGBA")
        
        # Patient head
        image_path = os.getcwd() + "\\apynf\\NF_model\\head.png"
        illustration = Image.open(image_path)
        ill_x,ill_y = illustration.size
        prop = 0.7
        new_x = int(prop*x_size)
        new_y = int((new_x/ill_x)*ill_y)
        illustration =  illustration.resize((new_x,new_y),Image.ANTIALIAS)
        pos_x = int(new_x/2.0)
        pos_y = int(y_size-1 - new_y/2.0)
        corners = centersize_to_box((pos_x,pos_y),(new_x,new_y))[0] + centersize_to_box((pos_x,pos_y),(new_x,new_y))[1]
        img.paste(illustration,corners)

        # Image borders
        draw.rectangle([(0,0),(x_size-1,y_size-1)],outline="black")

        # Mental 1D level
        mental_position = (int(0.1*x_size),int(0.5*y_size))
        mental_size = (int(0.1*x_size),int(0.75*y_size))
        self.mental.draw(draw,mental_position,mental_size)

        # # Subjective 1D level
        # subj_position = (int(0.4*x_size),int(0.5*y_size))
        # subj_size = (int(0.1*x_size),int(0.75*y_size))
        # self.subjective.draw(draw,subj_position,subj_size)

        # Screen
        screen_position = (int(0.9*x_size),int(0.5*y_size))
        screen_size = (int(0.1*x_size),int(0.75*y_size))
        self.screen.draw(draw,screen_position,screen_size)

        return img

        img.show()
    
    def show(self,size):
        self.draw(size).show()


if (__name__ == "__main__"):
    # model = NF_model(3)
    model = NF_model_displayer(5,5)
    model.show((1200,700))
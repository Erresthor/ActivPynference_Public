import sys,os
import numpy as np

from PIL import Image, ImageDraw, ImageFont 

from t_maze_test import generate_data

# A few functions to help us deal with images
def selective_transparency_shift(img,new_transparency):
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[3] == 255:
                newData.append((item[0],item[1],item[2],int(new_transparency*255)))
            elif item[3] == 0:
                newData.append(item)
            else : 
                newData.append((item[0],item[1],item[2],int(new_transparency*item[3])))
        img.putdata(newData)
        return img
        
def return_merge_im(im_a, im_b, xper , yper, alpha):
    widtha, heighta = im_a.size
    widthb, heightb = im_b.size

    im_foreground = im_b.copy()
    if (alpha < 0.999):
        im_foreground = selective_transparency_shift(im_foreground,alpha)
    # im_foreground.putalpha(int(255*alpha))
    im_return = im_a.copy()
    
    im_return.paste(im_foreground,[int(xper*widtha-0.5*widthb),int(yper*heighta-0.5*heightb)],im_foreground)
    # im_return.alpha_composite(im_foreground,[int(xper*widtha-0.5*widthb),int(yper*heighta-0.5*heightb)])#,im_foreground)
    return im_return

# Linear intepolation for arrays describing paths with 2+ waypoints (quick & dirty)
def get_current_pos(traj_arr,x):
    N_waypoints = traj_arr.shape[0]
    assert N_waypoints>1,"There should be at least 2 waypoints"
    prop_per_section = 1/(N_waypoints-1)
    
    x_section = int(x/prop_per_section)
    if (x_section>N_waypoints-2):
        x_section = N_waypoints-2 # For the last point
    # dx is a normalized value showing where we are in this sepecific section
    dx = (x - x_section*prop_per_section)/prop_per_section

    my_pos = traj_arr[x_section,:] + dx*(traj_arr[x_section+1,:]- traj_arr[x_section,:])
    return my_pos

class tmaze_drawer:
    """ 
    This class is used to plot the t-maze.
    One can use simulated data to plot animatins of the t maze at the trial / step scale.
    """ 
    def __init__(self,ressources_path):
        self.backg = Image.open(os.path.join(ressources_path, "background.png"))
        self.width,self.height = self.backg.size
        self.clue_left = Image.open(os.path.join(ressources_path, "clue_left.png"))
        self.clue_right= Image.open(os.path.join(ressources_path, "clue_right.png"))
        self.clue_unopened = Image.open(os.path.join(ressources_path, "clue_unopened.png"))

        self.mouse = Image.open(os.path.join(ressources_path, "mouse.png"))
        self.mousetrap = Image.open(os.path.join(ressources_path, "mousetrap.png"))
        self.cheese = Image.open(os.path.join(ressources_path, "cheese.png"))

    def get_step_mazeplot(self,NFRAMES,from_state,to_state,cheese_perception, clue="unopened",
                          trial=0,max_trial=0):
        LEFT_LINE_X = 0.2
        MID_LINE_X = 0.5
        RIGHT_LINE_X = 0.8
        
        TOP_LINE_Y = 0.15
        MID_LINE_Y = 0.5
        BOT_LINE_Y = 0.85

        # For the mouse position
        waypoint_dict = {
            "0" : [MID_LINE_X,MID_LINE_Y],
            "1" : [MID_LINE_X,BOT_LINE_Y],
            "2" : [LEFT_LINE_X,TOP_LINE_Y],
            "3" : [RIGHT_LINE_X,TOP_LINE_Y],
            "MID" : [MID_LINE_X,TOP_LINE_Y]
        }
        waypoint_list = []
        # 1st waypoint :
        waypoint_list.append(waypoint_dict[str(int(from_state))])
        # If we make the angle :

        pass_through_hinge = ((from_state in [0,1]) and (to_state in [2,3]))
        outside_scope = ((from_state in [2,3]) and (to_state in [0,1]))
        if outside_scope :
            return []
        if pass_through_hinge:
            NFRAMES = int(NFRAMES*2)
            waypoint_list.append(waypoint_dict["MID"])
        # last waypoint :
        waypoint_list.append(waypoint_dict[str(int(to_state))])
        # Make into array : 
        traj_array = np.array(waypoint_list)

        # Create the background for this step:
        # DRAW CLUE ON BACKGROUND :
        base_img = Image.new('RGBA', (self.width, self.height))
        base_img = return_merge_im(base_img,self.backg, 0.5,0.5,1.0)
        if (clue=="unopened"):
            base_img = return_merge_im(base_img, self.clue_unopened, MID_LINE_X,BOT_LINE_Y,1.0)
        elif (clue=="left"):
            base_img = return_merge_im(base_img, self.clue_left, MID_LINE_X,BOT_LINE_Y,1.0)
        elif (clue=="right"):
            base_img = return_merge_im(base_img, self.clue_right, MID_LINE_X,BOT_LINE_Y,1.0)
        
        # DRAW MOUSE PERCEPTION 
        base_img = return_merge_im(base_img, self.mousetrap, LEFT_LINE_X,TOP_LINE_Y,1-cheese_perception[0])
        base_img = return_merge_im(base_img, self.cheese, LEFT_LINE_X,TOP_LINE_Y,cheese_perception[0])

        base_img = return_merge_im(base_img, self.mousetrap, RIGHT_LINE_X,TOP_LINE_Y,1-cheese_perception[1])
        base_img = return_merge_im(base_img, self.cheese, RIGHT_LINE_X,TOP_LINE_Y,cheese_perception[1])

        draw = ImageDraw.Draw(base_img)
        text = str(trial) + " / " + str(max_trial)
        # font = ImageFont.load_default() # Load a font
        fontsize = 30
        font = ImageFont.truetype("arial.ttf", fontsize)

        rectangle_coords = (int(0.0*self.width), int(0.85*self.height), int(0.3*self.width), int(1.0*self.height))  # Coordonnées du coin supérieur gauche (x1, y1) et du coin inférieur droit (x2, y2)
        draw.rectangle(rectangle_coords, fill ="grey",outline="black", width=2)
        draw.text((int(0.05*self.width),int(0.89*self.height)),text,font=font,fill="black")
        
        imglist = []
        for t in np.linspace(0,1.0,NFRAMES):
            this_step_img = base_img.copy()
            mouse_pos = get_current_pos(traj_array,t)
            this_step_img = return_merge_im(this_step_img, self.mouse, mouse_pos[0],mouse_pos[1],1.0)
            imglist.append(this_step_img)
        return imglist

    def get_trial_mazeplot(self,
                trial,max_trial,Nframes,
                true_reward_state,true_agent_state,
                reward_state_perception,clue_observation,
                agent_post_act,agent_actions):
        total_gif = []
        T = true_reward_state.shape[0]
        

        cluestate = "unopened"
        for t in range(T):
            # First, a few frames to show the starting situation
            this_step_from_state = true_agent_state[t]
            cheese_perception = reward_state_perception[:,t]
            clue_observation_t = clue_observation[t] # 0 if none, 1 if L, 2 if R
            if (cluestate == "unopened"):
                if (clue_observation_t==1):
                    cluestate="left"
                if (clue_observation_t==2):
                    cluestate="right"               
            one_frame = self.get_step_mazeplot(1,this_step_from_state,this_step_from_state,cheese_perception,
                                               cluestate,trial,max_trial)
            for k in range(int(Nframes/2)):
                total_gif += one_frame
            # Then, if we aren't done, the various steps showing the agent action
            if (t<T-1):
                this_step_to_state = true_agent_state[t+1]
                total_gif = total_gif + self.get_step_mazeplot(
                        Nframes,this_step_from_state,this_step_to_state,
                        cheese_perception,cluestate,trial,max_trial)
        return total_gif
        # Finally, a few frames to show the resolution       

import random as ran

if __name__=="__main__":
    ressources_path = os.path.join("ressources/tmaze")
    drawer = tmaze_drawer(ressources_path)
    
    bad_imgs = []
    good_imgs = []
    for k in range(40):
        r = ran.randint(0,1)
        r2 = ran.randint(0,1)
        # Good clue :
        clueval = "right" if r == 0 else "left"
        imgl = drawer.get_step_mazeplot(5,1,0,[r,1-r], clue=clueval,
                            trial=0,max_trial=0)
        good_imgs.append(imgl[3])
        # bad clue :
        clueval = "right" if r2 == 0 else "left"
        imgl = drawer.get_step_mazeplot(5,1,0,[r,1-r], clue=clueval,
                            trial=0,max_trial=0)
        bad_imgs.append(imgl[3])
        # imgl[12].save(os.path.join(ressources_path,"point_right.png"))

        # total_gif[0].save("ressources/tmaze/renders/render.gif", save_all=True, append_images=total_gif[1:],duration=30,loop=0)
    
    good_imgs[0].save("ressources/tmaze/renders/goodclue.gif", save_all=True, append_images=good_imgs[1:],duration=400,loop=0)
    bad_imgs[0].save("ressources/tmaze/renders/badclue.gif", save_all=True, append_images=bad_imgs[1:],duration=400,loop=0)
    
    
    
    
    
    
    
    
    
    
    # # GENERATE A GIF :
    # total_gif = []
    # N = 30
    # generated_data = generate_data(30,pinit=0.5,pinit2=0.5,pHA=0.5,rs=2,la=1)
    # true_cheese_pos,true_agent_pos,reward_state_perception,agent_post_act,agent_actions,clue_obss = generated_data
    # for trial in range(N):
    #     print("Trial " + str(trial+1) + " / " + str(N))
    #     true_reward_state = true_cheese_pos[:,trial]
    #     true_agent_state = true_agent_pos[:,trial]
    #     cheese_perception = reward_state_perception[:,:,trial]
    #     clue_obs = clue_obss[:,trial]
    #     imglist = drawer.get_trial_mazeplot(trial+1,N,
    #                           15,true_reward_state,true_agent_state,
    #                           cheese_perception,clue_obs,
    #                           None,None)
    #     total_gif += imglist
    # total_gif[0].save("ressources/tmaze/renders/render.gif", save_all=True, append_images=total_gif[1:],duration=30,loop=0)



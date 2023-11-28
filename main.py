from demo.my_projects.internal_observator_hypothesis import get_nf_network,ind2sub,sub2ind
from actynf.base.function_toolbox import normalize
import numpy as np
from PIL import Image
import sys

if __name__ == '__main__':  
    my_net = get_nf_network(32,100)
    new_width = 400
    new_height = 400
    
    Ntrials = 100

    imglist = []
    maze_perception = []
    for t in range(Ntrials):
        my_net.run()
        arr = normalize(np.copy(my_net.layers[1].a[1]))
        image = Image.fromarray((255*arr).astype(int))
        # Resize the PIL image
        
        resized_pil_image = image.resize((new_width, new_height),resample=Image.NEAREST)
        imglist.append(resized_pil_image) #.show()
        print(my_net.layers[1].STM.u)
        print(np.round(my_net.layers[1].b[0][:,:,3],2))

        xds = my_net.layers[1].STM.x_d
        for t in range(xds.shape[1]):
            arr = np.zeros((4,8))
            for state in range(xds.shape[0]):
                state_x,state_y = ind2sub((4,4),state)
                arr[state_x,state_y] = xds[state,t]

            # print(my_net.layers[0].STM.x)
            true_x =  my_net.layers[0].STM.x[0,t]
            true_state_x,true_state_y = ind2sub((4,4),true_x)
            arr[true_state_x,true_state_y+4] = 1.0

            image = Image.fromarray((255*arr).astype(int))
            resized_pil_image = image.resize((new_width, new_height),resample=Image.NEAREST)
            maze_perception.append(resized_pil_image)

        image = Image.fromarray((255*np.ones((4,8))).astype(int))
        resized_pil_image = image.resize((new_width, new_height),resample=Image.NEAREST)
        maze_perception.append(resized_pil_image)

    imglist[0].save('learn_internal_observer.gif', save_all=True, append_images=imglist,duration=50, loop=0)
    maze_perception[0].save('cognitive_state_perception.gif', save_all=True, append_images=maze_perception,duration=300, loop=0)
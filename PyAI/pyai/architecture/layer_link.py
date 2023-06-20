import numpy as np 
from ..base.function_toolbox import isField,spm_complete_margin,spm_cross
from ..base.miscellaneous_toolbox import flexible_copy


def get_negative_range(axes,max_size):
    negative = tuple()
    for k in range(max_size):
        if not(k in axes):
            negative += (k,)
    return negative

def get_joint_distribution_along(distribution,axes):
    summed_distribution = np.sum(distribution,get_negative_range(axes,distribution.ndim))
    # Depending on the order of the axes, we swap the distribution
    # axes :
    ax_list = list(axes)

    n = len(ax_list)
    for i in range(n):
        for j in range(n - i - 1):
            if ax_list[j] > ax_list[j + 1]:
                # SWAP J AND J+1
                ax_list[j], ax_list[j + 1] = ax_list[j + 1], ax_list[j]
                # DO THE SAME FOR THE SUMMED_DISTRIBUTION
                summed_distribution = np.swapaxes(summed_distribution,j,j+1)
    return summed_distribution

def get_margin_distribution_along(distribution,axes):
    marginalized = spm_complete_margin(distribution)
    return_this = []
    if type(axes)==tuple:   
        for index in axes:
            return_this.append(marginalized[index])
        return return_this
    if type(axes)==int:
        return marginalized[axes]
    raise NotImplementedError('The type ' + type(axes) + ' is not implemented for this function')

def get_definite_value_along(array,axes):
    return_this = []
    if type(axes)==tuple:   
        for index in axes:
            return_this.append(array[index])
        return return_this
    if type(axes)==int:
        return array[axes]

def get_joint_values_from_array(array, along_dimensions, is_joint_distribution=False):
    if (along_dimensions==None):
        return array
    
    if (type(along_dimensions)==tuple):
        if (is_joint_distribution):
            return get_joint_distribution_along(array,along_dimensions)
        else :
            return array[np.array(along_dimensions)]

    if (type(along_dimensions)==int):
        if (is_joint_distribution):
            return get_joint_distribution_along(array,tuple(along_dimensions))
        else :
            return array[along_dimensions]

def transmit_data(from_object,from_field,from_axes,to_object,to_field,to_axes):
    """ 
    This function updates the stored values in the to_object using values in the from_object.
    More specifically, this function uses strings (from_fields and to_field) to 
    look for specific attributes of the from and to objects. The axes arguments 
    allow for a better control of the selected dimensions : 
    Important note regarding the axes parameters :
        - 3 types accepted : ints, tuples & None
        - to_axes and from_axes are obviously closely related regarding their shapes
        - from_axes is an integer iff to_axes is also an integer. 
        - If from axes is None, all the dimensions of from_field are selected.
        - If to_axes is None, all the dimensions of to_field are changed.
    Other notes :
        - This transmission scheme assumes conditionnal independence of transmitted data (from_object.from_field[from_axes])
        and data already in the to_object that aren't replaced (to_object.to_field[NOT to_axes]).
    """
    from_field_isDist = "d" in from_field
    to_field_isDist = "d" in to_field

    # Fetch data from the from_object
    from_value_full = getattr(from_object,from_field)
    if (not isField(from_value_full)):
        raise ValueError("Transmission impossible from field [" + str(from_field) + "] of the object\n" + str(from_object)+"--> field has not been initialized.")
    to_value = getattr(to_object,to_field)
    if (not isField(to_value)):
        raise ValueError("Transmission impossible to field [" + str(to_field) + "] of the object\n" + str(to_object)+"--> field has not been initialized.")
    

    # from_value_full and to_value are considered available at this point
    if not(isField(from_axes)):
        from_axes = tuple(range(from_value_full.ndim))
    if not(isField(to_axes)):
        to_axes = tuple(range(to_value.ndim))
    
    if (to_field_isDist):
        to_value = to_value
        to_value_shape = to_value.shape
        shape_of_new_input = list(to_value_shape)

        if (from_field_isDist):
            from_value = get_joint_distribution_along(from_value_full,from_axes)
        else :
            from_value = from_value_full[np.array(from_axes)]
            # TODO : Transform this from_value (1d array of ints)
            # Into a joint distribution with 1 where the realized outcome is
            raise NotImplemented("layerlink : Transmissions << definite ---> joint >> not implemented yet...")       
        
        # RESHAPE THE NEW DISTRIBUTION :
        # 1. Take this_range(len(joint2.shape))
        # 2.a Initialize the shape of the changed joint distribution shapeJoin
        # 2.b Iterate el in this_range :
        #       - If el is in axes_to_joint_distrib, then shapeJoin[el] is joint2.shape[el]
        #       - Else shapeJoin[el] is 1
        for i in range(len(to_value_shape)):
            if not(i in to_axes):
                shape_of_new_input[i] = 1
        reshape_of_new_input = tuple(shape_of_new_input)
        reshaped_from = np.reshape(from_value,reshape_of_new_input)
        reshaped_to = np.sum(to_value,to_axes,keepdims=True)

        updated_to_value = reshaped_from*reshaped_to
        assert (np.sum(updated_to_value)-1.0)<1e-10, "Error when updating the input for " + to_object.layer.name + " with the output of " + from_object.layer.name + ". The final distribution sums to " + str(np.sum(updated_to_value)) + " instead of 1."
        
        # Change the to_value field to the updated_to_value
        
    else :
        updated_to_value = flexible_copy(to_value)
        if (from_field_isDist):
            from_value = get_joint_distribution_along(from_value_full,from_axes)
            # TODO : Transform this from_value (n-dim joint distribution)
            # Into a set of indices using classical sampling ?
            raise NotImplemented("layerlink : Transmissions << joint ---> definite >> not implemented yet...")
        else :
            from_value = from_value_full[np.array(from_axes)]
        
        for k in range(len(to_axes)):
            updated_to_value[to_axes[k]] = from_value[k]
    
    setattr(to_object,to_field,updated_to_value) 
    return

def transmit_data_old(from_object,from_field,from_axes,to_object,to_field,to_axes):
    """ 
    This OLD transmission scheme assumes conditionnal independence of transmitted data (from_object.from_field[from_axes])
    and data already in the to_object that aren't replaced (to_object.to_field[NOT to_axes])
    AS WELL AS conditonnal independence between variables of the same from_field. A new version is available
    to correct that.

    -----
    This function updates the stored values in the to_object using values in the from_object.
    More specifically, this function uses strings (from_fields and to_field) to 
    look for specific attributes of the from and to objects. The axes arguments 
    allow for a better control of the selected dimensions : 
    Important note regarding the axes parameters :
        - 3 types accepted : ints, tuples & None
        - to_axes and from_axes are obviously closely related regarding the shapes
        - from_axes is an integer iff to_axes is also an integer. 
        - If from axes is None, all the dimensions of from_field are selected.
        - If to_axes is None, all the dimensions of to_field are changed.
    """
    from_field_isDist = "d" in from_field
    to_field_isDist = "d" in to_field

    from_value_full = getattr(from_object,from_field)
    to_value = getattr(to_object,to_field)

    # from_value_full and to_value are considered available at this point
    if not(isField(from_axes)):
        from_axes = range(from_value_full.ndim)
    if not(isField(to_axes)):
        to_axes = range(to_value.ndim)

    if (from_field_isDist and to_field_isDist):
        from_value = get_joint_distribution_along(from_value_full,from_axes)
    else :
        from_value = from_value_full[np.array(from_axes)]
    to_value = to_value


    # IF FROM VALUE IS A DIST AND TO VALUE IS A DIST : 
    to_value_shape = to_value.shape
    negative_to_axes = get_negative_range(to_axes,to_value.ndim)

    shape_of_new_input = list(to_value_shape)
    # RESHAPE THE NEW DISTRIBUTION :
    # 1. Take this_range(len(joint2.shape))
    # 2.a Initialize the shape of the changed joint distribution shapeJoin
    # 2.b Iterate el in this_range :
    #       - If el is in axes_to_joint_distrib, then shapeJoin[el] is joint2.shape[el]
    #       - Else shapeJoin[el] is 1
    for i in range(len(to_value_shape)):
        if not(i in to_axes):
            shape_of_new_input[i] = 1
    reshape_of_new_input = tuple(shape_of_new_input)
    
    
    reshaped_from = np.reshape(from_value,reshape_of_new_input)
    reshaped_to = np.sum(to_value,to_axes,keepdims=True)
    updated_to = reshaped_from*reshaped_to

    # If from_axes AND to_axes are ints or len 1 tuples, 
    # 1 / make them len 1 tuples
    # 2/  the transmission can be done with the marginalized values only :
    if (type(to_axes)==int):
        to_axes = (to_axes,)
        assert (type(from_axes)==int)or((type(from_axes)==tuple)and(len(from_axes)==1)),"from_axes and to_axes should have the same dimension."
        if (type(from_axes)==int):
            from_axes = (from_axes,)

        if (from_field_isDist):
            from_dists = get_margin_distribution_along(from_value,from_axes)
        else :
            from_dists = get_definite_value_along(from_value,from_value)

        if (to_field_isDist):
            new_to_value = spm_complete_margin(to_value)
        else :
            new_to_value = to_value

        for k in range(len(to_axes)):
            new_to_value[to_axes[k]] = from_dists[k]
        if (to_field_isDist):  # Make the marginalized distribution list a joint distribution by assuming variable independence
            new_to_value = spm_cross(new_to_value)
        setattr(to_object,to_field,new_to_value)
    
    else :
        # If (from_axes OR to_axes) are (None or tuple) transmit the whole joint distribution

        # Transform the from fields into marginal distributions and/or slice the required input values
        if (from_field_isDist):
            from_dists = get_joint_distribution_along(from_value,from_axes)
        else :
            from_dists = get_definite_value_along(from_value,from_value)

        new_to_value = flexible_copy(to_value)
        
        for k in range(len(to_axes)):
            new_to_value[to_axes[k]] = from_dists[k]

        
        setattr(to_object,to_field,new_to_value)
        print("Done !")
        print(from_object)
        print(to_object)
        

class layerLink:
    """ 
    A layerLink establishes a data pipeline between a single output and a single input
    of two layers.
    """
    def __init__(self,from_out,to_in,list_of_connections=None):
        self.from_output = from_out
        self.to_input = to_in
        self.connections = []

        # If a list of connections has been provided in the constructor, 
        # initialize the following connections
        if (isField(list_of_connections)):
            assert type(list_of_connections)==list,"list_of_connections should be a list of list of strings."
            for connection_demand in list_of_connections:
                assert type(connection_demand)==list,"list_of_connections should be a list of list of strings."
                self.connect(connection_demand[0],connection_demand[1])
        # transmit data from : 
        #       0 : raw observations if None, closed pipeline /// if tuple, modalities (ki,kj, etc)
        #       1 : raw states if None, closed pipeline /// if tuple, factors (ki,kj, etc)
        #       2 : raw actions if None, closed pipeline /// if tuple or int AND value = 0, the raw action
        #       3 : observation distributions if None, closed pipeline /// if tuple, modalities (ki,kj, etc)
        #       4 : state distributions if None, closed pipeline /// if tuple, factors (ki,kj, etc)#       0 : raw observations if None, closed pipeline /// if tuple, modalities (ki,kj, etc)
        #       5 : actions distributions if None, closed pipeline /// if tuple or int AND value = 0, the posterior over actions / action distributions

    def generate_empty_mapping_dictionnary(self):
        No = self.from_output.layer.No
        Ns = self.from_output.layer.Ns
        mapping_dict = {
        }
        # u & u_d:
        mapping_dict["u"] = None
        mapping_dict["qu"] = None
        # O & o_d: 
        for mod in range(len(No)):
            mapping_dict["o"][str(mod)] = None
            mapping_dict["o_d"][str(mod)] = None
        # S & s_d:
        for fac in range(len(Ns)):
            mapping_dict["s"][str(fac)] = None
            mapping_dict["s_d"][str(fac)] = None
        return mapping_dict 

    def direct_link(self):
        """ If from_output and to_input are layers of 
        same observation & state shapes, we can link them directly :"""
        
        assert self.out_Ns==self.in_Ns , "A direct link was created between " + self.from_output.layer.name + " and " + self.to_input.layer.name + " but hidden state dimensions don't match. Please specify a filter to connect specific factors only."
        assert self.out_No==self.in_No , "A direct link was created between " + self.from_output.layer.name + " and " + self.to_input.layer.name + " but observations dimensions don't match. Please specify a filter to connect specific modalities only."
        
        direct_link_dict = self.generate_empty_mapping_dictionnary()
        # u & u_d :
        direct_link_dict["u"] = "u"
        direct_link_dict["qu"] = "u_d"
        # O & o_d:
        for mod in range(len(self.out_Ns)):
            code_o = "o" + "." + str(mod)
            direct_link_dict["o"][str(mod)] = code_o
            code_o_d = "o_d" + "." + str(mod)
            direct_link_dict["o_d"][str(mod)] = code_o_d
        # S & s_d:
        for fac in range(len(self.out_No)):
            code_s = "s" + "." + str(fac)
            direct_link_dict["s"][str(fac)] = code_s
            code_s_d = "s_d" + "." + str(fac)
            direct_link_dict["s_d"][str(fac)] = code_s_d
        return direct_link_dict

    def check_if_parrallel_paths(self):
        """ 
        Check if in the connection list there exist two 
        paths with the same from field and to the same field
        If it is the case, we merge those.
        """
        for connection_A_idx in range(len(self.connections)) :
            # If we manage to find a connection that ressembles connection_A
            # among all connections except A : 
            for connection_B_idx in [i for i in range(len(self.connections)) if i!=connection_A_idx] :
                conn_A = self.connections[connection_A_idx]
                conn_B = self.connections[connection_B_idx]
                if (conn_A[0][0]==conn_B[0][0]) and (conn_A[1][0]==conn_B[1][0]):
                    print("Found a parrallel connection on " + conn_A[0][0] + " --> " + conn_B[1][0] + " .")
                    print("Merging ...")
                    from_conns = conn_A[0][1]+conn_B[0][1]
                    to_conns = conn_A[1][1]+conn_B[1][1]
                    new_self_connections = self.connections
                    self.connections[connection_A_idx] = [[conn_A[0][0],from_conns],[conn_A[1][0],to_conns]]
                    self.connections.pop(connection_B_idx)
                    print("Merge successful !")
                    return self.check_if_parrallel_paths()
        return
    
    def connect(self,from_code, to_code):
        """ 
        THIS CODE COMPARES DIMENSIONS ACROSS FROM AND TO LAYERS AND CHECK THAT A CONNECTION IS POSSIBLE
        From code & to code are either of form 'a.k' or of form 'a' 
        with a :
            - 'u','s'or'o'  + "_d" OR NOT
        and k :
            - an integer if we connect just a dimension
            - "k1-k2" if we want to connect specific dimensions (doesn't have to be ascending order)
            - Nothing if we want to connect all dims

        """
        if not("." in from_code) :
            from_code = from_code+".ALL"
        if not("." in to_code) :
            to_code = to_code+".ALL"
        
        dimension_from = from_code.split(".")[0]
        mod_from = from_code.split(".")[1]
        dimension_to = to_code.split(".")[0]
        mod_to = to_code.split(".")[1]

        if (mod_to==""):
            mod_to = "ALL"
        if (mod_from==""):
            mod_from="ALL"

        assert ("o" in dimension_from) or ("u" in dimension_from) or ("s" in dimension_from),dimension_from + " is an invalid connection prompt."
        assert ("o" in dimension_to) or ("u" in dimension_to) or ("s" in dimension_to),dimension_to + " is an invalid connection prompt."

        dim_letter_from =  dimension_from
        dim_letter_to = dimension_to
        if ("_" in dimension_from):
            dim_letter_from = dimension_from.split("_")[0]
        if ("_" in dimension_to):
            dim_letter_to = dimension_to.split("_")[0]
        if (dim_letter_from=="u"):
            dim_letter_from = "p"
        if (dim_letter_to=="u"):
            dim_letter_to ="p"

        total_size_from = getattr(self.from_output.layer,"N" + dim_letter_from)
        total_size_to = getattr(self.to_input.layer,"N" + dim_letter_to)

        print(total_size_from,total_size_to)

        if(mod_from=="ALL"):
            compare_from = total_size_from
            from_axes = tuple(range(len(total_size_from)))
        else :
            if ("-" in mod_from):
                mod_from = mod_from.split("-")
                mod_from = [int(x) for x in mod_from]
                from_axes = tuple(mod_from)
                compare_from = [total_size_from[i] for i in mod_from]
            else:
                compare_from = total_size_from[int(mod_from)]
                from_axes = (int(mod_from),)



        if(mod_to=="ALL"):
            compare_to = total_size_to
            to_axes = tuple(range(len(total_size_to)))
        else :
            if ("-" in mod_to):
                mod_to = mod_to.split("-")
                mod_to = [int(x) for x in mod_to]
                print(total_size_to)
                print(np.array(mod_to))
                compare_to = [total_size_to[i] for i in mod_to]
                to_axes = tuple(mod_to)
                
            else:
                compare_to = total_size_to[int(mod_to)]
                to_axes = (int(mod_to),)


        assert compare_from==compare_to,"Can't connect " + from_code + " (layer : " + self.from_output.layer.name + " ) and "  +  to_code +" (layer : " + self.to_input.layer.name + " ) : dimensions don't match ( " + str(compare_from) + " =/= " + str(compare_to) + " )."

        # connection between [[from_attribute,from_attribute_modality]] and [[to_attribute,to_attribute_modality]]
        self.connections.append([[dimension_from,from_axes],[dimension_to,to_axes]])
        self.check_if_parrallel_paths()
        return 
    
    def fire_all_connections(self):
        """
        Transmit data along all the predefined transmissions paths.
        """
        for connection in self.connections:
            from_field = connection[0][0]
            from_axes = connection[0][1]
            to_field = connection[1][0]
            to_axes = connection[1][1]
            transmit_data(self.from_output,from_field,from_axes,self.to_input,to_field,to_axes)
import numpy as np 
from ..base.function_toolbox import isField,spm_complete_margin,spm_cross
from ..base.miscellaneous_toolbox import flexible_copy
from .utils import check_prompt_shape,get_joint_distribution_along

from .layer_components import layer_output,layer_input

# TODO : one should be able to merge two or more links between 
# the same two layers.
    
def axes_from_list(my_list):
    from_axes = tuple()
    to_axes = tuple()
    for from_to_couple in my_list:
        from_axes += (from_to_couple[0],)
        to_axes += (from_to_couple[1],)
    return from_axes,to_axes

def has_shared_layerLink(a,b):
    """ 
    Check if there is a link between two layer
    If there is, returns a list of the corresponding links.
    """
    item_list = []
    for item in a.outputs.links:
        if item in b.inputs.links:
            item_list.append(item)
    return len(item_list)>0,item_list

def get_layerLinks_between(a,b):
    isConnF,forward_link = has_shared_layerLink(a,b)
    isConnB,backward_link = has_shared_layerLink(b,a)
    return_those_links = []
    return_those_links.append(forward_link)
    return_those_links.append(backward_link)
    return return_those_links

def str_layerLinks_between(a,b):
    link_list = get_layerLinks_between(a,b)
    my_str = "Links between " + a.name + " and " + b.name + " :\n"
    my_str += "------ FORWARD : ------\n"
    for forward_conn in link_list[0]:
        my_str+= "   " + str(forward_conn)
    if (len(link_list[0])==0):
        my_str+= "   " +"NONE\n"
    my_str+= "------ BACKWARD : ------\n"
    for backward_conn in link_list[1]:
        my_str+= "   " +str(backward_conn)
    if (len(link_list[1])==0):
        my_str+= "   " +"NONE\n"
    return '\t'.join(my_str.splitlines(True))

def establish_layerLink(from_object,to_object,
                        list_of_connections=None,verbose=True,merge_verbose=True,auto_merge=True):
    if(type(from_object)!=layer_output):
        # Assume this object has a layer_output attribute
        try :
            from_object = from_object.outputs
        except : 
            raise TypeError("Object : \n" + str(from_object) + " should either be a "+str(layer_output)+" object or have a "+str(layer_output)+"  attribute, but is " + str(type(from_object)))
    if(type(to_object)!=layer_input):
        # Assume this object has a layer_input attribute
        try :
            to_object = to_object.inputs
        except : 
            raise TypeError("Object : \n" + str(to_object) + " should either be a "+str(layer_input)+" object or have a "+str(layer_input)+" attribute, but is " + str(type(to_object)))
    
    is_link_exist, existing_links = has_shared_layerLink(from_object.parent, to_object.parent)
    if (is_link_exist):
        print("/!\ There is already a layerLink from " + from_object.parent.name + " to " + to_object.parent.name +". Adding a new connection instead ...")
        assert len(existing_links) == 1, "There should only be a single layerLink here... something went wrong :(."
        existing_links[0].connect_list(list_of_connections,merge_verbose,auto_merge)
        return existing_links[0]
    
    # if (check_duplicate_links(from_out,to_in)):
        #     print("There is already a layerLink from " + from_out.parent.name + " to " + to_in.parent.name +".Adding a new connection instead ...")
        #     print("THIS IS A DUPLICATE ! REEEEEEEEEEEEEEE")
    return layerLink(from_object,to_object,
                        list_of_connections,verbose,merge_verbose,auto_merge)

class linkConnection:
    def __init__(self,
                 from_o,from_f,from_axs,
                 to_o,to_f,to_axs):
        self.from_object = from_o
        self.from_field = from_f
        self.from_axes = from_axs

        self.to_object = to_o
        self.to_field = to_f
        self.to_axes = to_axs

    def __str__(self):
        return  "Connection between " + str(self.from_object.parent.name) + " and " + str(self.to_object.parent.name) + " : " + self.get_field_axes_str()
    
    def get_field_axes_str(self):
        return_this = str(self.from_field) + " " + str(self.from_axes) + " --> "
        return_this += str(self.to_field) + " " + str(self.to_axes) + ""
        return  return_this

    def is_same_target_linkConnection(self,other_linkConnection):
        same_object = (other_linkConnection.from_object == self.from_object)and(other_linkConnection.to_object==self.to_object)
        same_fields = (other_linkConnection.from_field == self.from_field)and(other_linkConnection.to_field==self.to_field)
        return same_object and same_fields

    def is_same_linkConnection(self,other_linkConnection):
        same_axes = (other_linkConnection.from_axes == self.from_axes)and(other_linkConnection.to_axes==self.to_axes)
        return self.is_same_target_linkConnection(other_linkConnection) and same_axes

    def remove_duplicates(self):
        my_fromtos = [[self.from_axes[k],self.to_axes[k]] for k in range(len(self.from_axes))]
        
        newlist = [] # empty list to hold unique elements from the list
        duplist = [] # empty list to hold the duplicate elements from the list
        for i in my_fromtos:
            if i not in newlist:
                newlist.append(i)
            else:
                duplist.append(i)
                print("Found duplicate element " + str(i) + " . Removing ...")
        new_from_axes,new_to_axes = axes_from_list(newlist)
        self.from_axes = new_from_axes
        self.to_axes = new_to_axes

    def attempt_merge(self,other_linkConnection):
        """ Return a merged version of the two connections, where duplicate pipelines have been removed."""
        if (not(self.is_same_target_linkConnection(other_linkConnection))):
            return False
        my_fromtos = [[self.from_axes[k],self.to_axes[k]] for k in range(len(self.from_axes))]
        other_fromtos = [[other_linkConnection.from_axes[k],other_linkConnection.to_axes[k]] for k in range(len(other_linkConnection.from_axes))]
        for myft_idx in range(len(my_fromtos)) :
            myft = my_fromtos[myft_idx]
            if (myft in other_fromtos):
                my_fromtos.pop(myft_idx)
        new_from_axes,new_to_axes = axes_from_list(my_fromtos + other_fromtos)
        self.from_axes = new_from_axes
        self.to_axes = new_to_axes
        self.remove_duplicates()
        return True # other_linkConnection should be removed
    
    def transmit(self):
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
        from_field_isDist = "d" in self.from_field
        to_field_isDist = "d" in self.to_field

        # Fetch data from the from_object
        from_value_full = getattr(self.from_object,self.from_field)
        if (not isField(from_value_full)):
            raise ValueError("Transmission impossible from field [" + str(self.from_field) + "] of the object\n" + str(self.from_object)+"--> field has not been initialized.")
        
        # from_value_full and to_value are considered available at this point
        from_axes = (tuple(range(from_value_full.ndim)) if not(isField(self.from_axes)) else self.from_axes)

            
        if (to_field_isDist):
            if (from_field_isDist):
                from_value = get_joint_distribution_along(from_value_full,from_axes)
            else :
                from_value = from_value_full[np.array(from_axes)]
                # TODO : Transform this from_value (1d array of ints)
                # Into a joint distribution with 1 where the realized outcome is
                raise NotImplemented("layerlink : Transmissions << definite ---> joint >> not implemented yet...")    
            self.to_object.update_distribution_input(self.to_field,self.to_axes,from_value)
        else :
            if (from_field_isDist):
                from_value = get_joint_distribution_along(from_value_full,from_axes)
                # TODO : Transform this from_value (n-dim joint distribution)
                # Into a set of indices using classical sampling ?
                raise NotImplemented("layerlink : Transmissions << joint ---> definite >> not implemented yet...")
            else :
                from_value = from_value_full[np.array(from_axes)]
            self.to_object.update_definite_input(self.to_field,self.to_axes,from_value)
  
class layerLink:
    """ 
    A layerLink establishes a data pipeline between a layer output and a layer input
    of two layers (or the same).
    """
    # INITIALIZE & COPY
    def __init__(self,from_out, to_in,
                list_of_connections=None,verbose=True,merge_verbose=True,auto_merge=True):               
        self.from_output = from_out
        self.to_input = to_in
        self.connection_prompts = [] 
            # A history of the user prompts that led to successful connections
            # Used when copying this link
            # WARNING : may not be the same size as self.connections (merging)
        self.connections = []

        # Let's inform the from_out and to_in that we are connecting !
        self.from_output.links.append(self)
        self.to_input.links.append(self)

        # If a list of connections has been provided in the constructor, 
        # initialize the following connections
        if (isField(list_of_connections)):
            self.connect_list(list_of_connections,merge_verbose,auto_merge)
        if(verbose):
            print("Established layerLink between " + str(from_out.parent.name) + " and " + str(to_in.parent.name) + ".") 

    def manual_copy(self,new_from_out,new_to_in,verbose=False,merge_verbose=False):
        """ 
        Establishes the same connections between two new layers
        Probably useless, deepcopy works better
        """
        return layerLink(new_from_out,new_to_in,flexible_copy(self.connection_prompts),verbose=False,merge_verbose=merge_verbose)

    # COSMETIC 
    def __str__(self):
        my_str = ""
        my_str += "Layer link << " + self.from_output.parent.name + " ---> " + self.to_input.parent.name + "  >> along field" + ("s" if (len(self.connection_prompts)>1) else "") + " : \n"
        for conn in self.connections :
            my_str += "    + " + conn.get_field_axes_str()
        return my_str
    
    # ESTABLISH / SEVER CONNECTIONS :
    def check_merge_connections(self,verbose=True):
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
                old_str_A = str(conn_A)
                if (conn_A.attempt_merge(conn_B)):
                    if(verbose):
                        print("   * Automatic merge between : \n       - " + old_str_A + "\n       - " + str(conn_B))
                        print("       == Resulting connection : " + str(conn_A))
                        print("   ((If this is unwanted behaviour, deactivate auto_merge in the layerLink builder.))")
                    # There was a successful merging 
                    self.connections.pop(connection_B_idx)
                    return self.check_merge_connections()
        return
    
    def connect_list(self, list_of_connection_prompts,verbose=True,auto_merge=True):
        assert type(list_of_connection_prompts)==list,"list_of_connections should be of form ['a','b'] or [['a1','b1'],['a2','b2']]."
        for connection_demand in list_of_connection_prompts:
            if not(type(connection_demand)==list):
                self.connect(list_of_connection_prompts[0],list_of_connection_prompts[1],verbose,auto_merge)
                return 
            else :
                self.connect(connection_demand[0],connection_demand[1],verbose,auto_merge)

    def connect(self,from_code, to_code,verbose=True,auto_merge=True):
        """ 
        THIS CODE COMPARES DIMENSIONS ACROSS FROM AND TO LAYERS AND CHECK THAT A CONNECTION IS POSSIBLE
        From code & to code are either of form 'a.k' or of form 'a' 
        with a :
            - ('u' or 's' or 'o')  + ("_d" OR nothing)
        and k :
            - an integer if we connect just a dimension
            - "k1-k2" if we want to connect specific dimensions (doesn't have to be ascending order)
            - Nothing if we want to connect all dims

        """
        from_code_extract = from_code+(".ALL" if not("." in from_code) else "")
        to_code_extract = to_code+(".ALL" if not("." in to_code) else "")
        
        dimension_from = from_code_extract.split(".")[0]
        dimension_to = to_code_extract.split(".")[0]

        mod_from = from_code_extract.split(".")[1]
        if (mod_from==""):
            mod_from="ALL"
        
        mod_to = to_code_extract.split(".")[1]
        if (mod_to==""):
            mod_to = "ALL"
        
        total_size_from = check_prompt_shape(dimension_from,self.from_output.parent)
        total_size_to = check_prompt_shape(dimension_to,self.to_input.parent)
        if(mod_from=="ALL"):
            compare_from = total_size_from
            if (type(total_size_from)==int):
                from_axes = (0,)
                # from_axes = (total_size_from,)
            else :
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
            if (type(total_size_to)==int):
                to_axes = (0,)
                # to_axes = (total_size_to,)
            else :
                to_axes = tuple(range(len(total_size_to)))
        else :
            if ("-" in mod_to):
                mod_to = mod_to.split("-")
                mod_to = [int(x) for x in mod_to]
                compare_to = [total_size_to[i] for i in mod_to]
                to_axes = tuple(mod_to)
            else:
                compare_to = total_size_to[int(mod_to)]
                to_axes = (int(mod_to),)
        assert compare_from==compare_to,"Can't connect " + from_code + " (layer : " + self.from_output.parent.name + " ) and "  +  to_code +" (layer : " + self.to_input.parent.name + " ) : dimensions don't match ( " + str(compare_from) + " =/= " + str(compare_to) + " )."

        # Create a new connection in the layer link
        # self.connections.append([[dimension_from,from_axes],[dimension_to,to_axes]])
        self.connections.append(linkConnection(self.from_output,dimension_from,from_axes,self.to_input,dimension_to,to_axes))
        self.connection_prompts.append([from_code,to_code])
        if (auto_merge):
            self.check_merge_connections(verbose)
        else :
            if (verbose):
                connections_str = ''
                for conn in self.connections :
                    connections_str += "                 -" + str(conn) + " \n"
                print("     --> Warning : No input field connections were merged for connections : \n" +connections_str+ "          ((If this is unwanted behaviour, activate auto_merge in the layerLink builder.))")
        return 
    
    def sever_connection(self,from_code, to_code):
        """ 
        Splices the input codes like the "connect" function.
        But instead tries to remove the corresponding connection(s)
        """
        raise NotImplementedError("Can't sever connection : function not yet implemented.")
  
    # TRANSMIT ACTUAL DATA
    def fire_all(self,verbose=False):
        """
        Transmit data along all the predefined transmissions paths.
        """
        if (len(self.connections) == 0):
            raise RuntimeError("No connections were specified for the layerLink [[\n " + str(self) + " ]].Can't transmit any data.")
        
        for connection in self.connections:
            connection.transmit()
            if(verbose):
                print("Transmitted data through connection : " + str(connection))

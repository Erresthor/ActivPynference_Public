class model_dimensions :
    def __init__(self) :
        self.T = None

        self.Nf = None
        self.Ns = None

        self.Nmod = None
        self.No = None

        self.Np = None
        self.Nu = None

class generative_model_variables :
    def __init__(self):
        self.a = None
        self.a_prior = None
        self.a_novelty = None
        self.a_ambiguity = None

        self.b = None
        self.b_prior = None
        self.b_complexity = None
        self.b_kron = None
        self.b_complexity = None
        
        self.c = None
        self.c_prior = None
        
        self.d = None
        self.d_prior = None
        self.d_complexity = None
        
        self.e = None
        self.e_prior = None

class layer_cache :
    def __init__(self):
        self.dims = model_dimensions()
        self.vars = generative_model_variables()
    
    def clear(self, keep_dims = False):
        if (keep_dims):
            temp_dims = self.dims
        self = layer_cache()
        if (keep_dims):
            self.dims = temp_dims
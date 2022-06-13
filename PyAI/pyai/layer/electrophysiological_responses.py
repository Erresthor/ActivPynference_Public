from ..base.function_toolbox import normalize
import numpy as np

def generate_electroph_responses(layer):
    n = 16
    points = np.arange(0,n,1)

    # DOPAMINE LEVELS

    h = np.exp(-points/2.0)
    h = h/np.sum(h)
    wn = np.kron(layer.w,np.ones((n,)))
    filter = np.concatenate((np.zeros(h.shape),h))
    conv_wn = np.convolve(wn,filter,'full')
    manual_cut = (filter.shape[0]-1) - int((filter.shape[0]-1)*0.5)
    conv_wn = conv_wn[manual_cut:wn.shape[0]+manual_cut]
    # Matlab and python convolve methods have an indexing problem -_-

    dn = np.gradient(conv_wn)

    
    return conv_wn,dn
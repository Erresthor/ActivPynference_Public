import statistics as stat
import numpy as np
from pyai.model.metrics import flexible_kl_dir
from pyai.base.function_toolbox import normalize
from pyai.new_layer.generative_process import GenerativeProcess,GenerativeProcessType

genproc = GenerativeProcess(GenerativeProcessType.MDP)
generator = genproc.get_generator()

print(generator(1,5))
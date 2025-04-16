# Denis
# -*- coding: utf-8 -*-
from .visual_quality import *
from .blur import *
from .occlusion import *
from .compress import *
from .temporal import *
from .nature import *
from .scene_interference import *
from .digital_process import *

seed = 42
torch.manual_seed(seed)
random.seed(seed)   
np.random.seed(seed) 
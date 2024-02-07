# -*- coding: utf-8 -*-
"""run_node_ablation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tbF5vCCyl3ew5nzQTF-rE52lkVeRBxv1

# Setup

## Install and Import Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install git+https://github.com/neelnanda-io/TransformerLens.git

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
# import plotly.express as px
from torch.utils.data import DataLoader

from jaxtyping import Float, Int
from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

import pickle
from google.colab import files

import matplotlib.pyplot as plt
import statistics

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

"""We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training."""

torch.set_grad_enabled(False)

"""## Load Model"""

model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)

"""## Import functions from repo"""

!git clone https://github.com/wlg1/seqcont_circ_expms.git

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/seqcont_circ_expms

from dataset import Dataset
from generate_data import *
from metrics import *
from head_ablation_fns import *
from mlp_ablation_fns import *
from node_ablation_fns import *
from loop_node_ablation_fns import *

"""# Run script"""

!python run_node_ablation.py --model "gpt2-small"
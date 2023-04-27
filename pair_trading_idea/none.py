import pandas as pd
import numpy as np
import tqdm
from itertools import combinations
import ray
from statsmodels.tsa.stattools import coint as eg_coint
import psutil
import os
from time import perf_counter

from fx_analysis.data import DataProvider
import matplotlib.pyplot as plt

from typing import *
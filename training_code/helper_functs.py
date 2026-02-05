# import stuff here before definitions
import matplotlib.pyplot as plt
import numpy as np
import math


# HELPER FUNCTION: Extract sorted unit names from all_data
def get_unit_names(data):
    """
    Extract and return sorted unit names from the all_data structure.
    Gets them from the 'rew1' condition (all conditions have the same units).
    """
    return sorted(data['rew1'].keys(), key=lambda x: int(x.split('_')[1]))
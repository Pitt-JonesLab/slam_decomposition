import logging
import pickle
import sys
from hashlib import sha1

import h5py
import numpy as np
from src import basis

#pickle only works if class definition must live in same module as when the object was stored
#use this trick so modules sees src.basis as basis
sys.modules['basis'] = basis

"""Loading/Saving helper functions"""

def filename_encode(arg):
    hash = sha1(arg.encode()).hexdigest() 
    return f"/home/evm9/decomposition_EM/data/{hash}.pkl"

def pickle_load(filename):
    """load a dictionary"""
    #try to open file if it exists
    try:
        with open(filename, 'rb') as f:
            loaded_data = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        loaded_data = {}

    return loaded_data

def pickle_save(filename, data_dict):
    """save a dictionary"""
    #at end, save back data dict
    logging.info("Saving data back to file")
    with open(filename, 'wb+') as f:
        pickle.dump(data_dict, f)


#XXX filename deprecated for h5py, update to work like pickle does above
def h5py_load(filekey, *args):
    """load a numpy array"""
    filename = f"data/{filekey}.h5"
    results = {}
    try:
        with h5py.File(filename, "r") as h5f:
            for arg in args:
                results[arg] = h5f[arg][:]
        return results
    except FileNotFoundError:
        logging.debug(f"Failed to load {filename}")
        return None

def h5py_save(filekey, **kwargs):
    """save a dictionary"""
    filename = f"data/{filekey}.h5"
    with h5py.File(filename, "a") as h5f:
        for key, value in kwargs.items():
            try:
                del h5f[key]
            except Exception:
                pass
            h5f.create_dataset(key, data=value)
    logging.debug(f"Successfully saved to {filename}")

#DEPRECRATED but might be useful snippets to have around
def rag_to_pad(arr):
    max_len = max(len(arr[i]) for i in range(len(arr)))
    for i in range(len(arr)):
        temp_len = len(arr[i])
        for j in range(max_len):
            if j >= temp_len:
                if j == 0:
                    raise ValueError("cant extend blank row")
                arr[i].append(arr[i][j - 1])
                # arr[i].append(-1)
    return np.array(arr)

# rewrite convert to ragged array by detecting when row is being extended
def pad_to_rag(arr):
    arr = arr.tolist()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            # if arr[i][j] == -1:
            if arr[i][j] == arr[i][j - 1]:
                arr[i] = arr[i][0:j]
                break
    return arr

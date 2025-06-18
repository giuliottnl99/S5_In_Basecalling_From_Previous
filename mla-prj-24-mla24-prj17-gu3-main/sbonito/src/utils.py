"""Contains general utilities
"""
import zipfile
import numpy as np
import signal
from contextlib import contextmanager
import torch.nn as nn 
def read_metadata(file_name):
    """Read the metadata of a npz file
    
    Args:
        filename (str): .npz file that we want to read the metadata from
        
    Returns:
        (list): with as many items as arrays in the file, each item in the list
        is filename (within the zip), shape, fortran order, dtype
    """
    zip_file=zipfile.ZipFile(file_name, mode='r')
    arr_names=zip_file.namelist()

    metadata=[]
    for arr_name in arr_names:
        fp=zip_file.open(arr_name,"r")
        version=np.lib.format.read_magic(fp)

        if version[0]==1:
            shape,fortran_order,dtype=np.lib.format.read_array_header_1_0(fp)
        elif version[0]==2:
            shape,fortran_order,dtype=np.lib.format.read_array_header_2_0(fp)
        else:
            print("File format not detected!")
        metadata.append((arr_name,shape,fortran_order,dtype))
        fp.close()
    zip_file.close()
    return metadata

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

def print_architecture(model):
    """Prints the architecture and important information about the model."""

    print("Model Architecture:\n")

    # Helper function to print individual layers or iterable modules
    def print_layers(name, module):
        print(f"{name}:")
        if module is None:
            print("  Not defined.")
        elif isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            for layer in module:
                print(f"  {layer}")
        else:
            print(f"  {module}")

    # Print Convolution Layers
    print_layers("Convolution Layers", getattr(model, 'convolution', None))

    # Print Encoder Layers
    print_layers("Encoder Layers", getattr(model, 'encoder', None))

    # Print Decoder Layers
    print_layers("Decoder Layers", getattr(model, 'decoder', None))

    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n--- Model Summary ---")
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    # Additional Information
    print("\nAdditional Information:")

    # Check if the model is on GPU or CPU
    device = next(model.parameters()).device
    print(f"Device: {device}")

    # Input and Output Size
    if hasattr(model, 'forward'):
        print("Input/Output shapes for each layer may vary based on input data.")
    else:
        print("No forward method found. Unable to determine input/output shapes.")

    # Number of layers
    num_layers = len(list(model.children()))
    print(f"Total Number of Layers: {num_layers}")

    print("\n--- End of Architecture ---")




class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    """Raise a TimeoutException if a function runs for too long

    Args:
        seconds (int): amount of max time the function can be run

    Example:
    try:
        with time_limit(10):
            my_func()
    except TimeoutException:
        do_something()
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
import hashlib
import random
import string
import time
import multiprocessing
import numpy as np
from numba import cuda

# Number of GPUs available
NUM_GPUS = 9  # Change this based on your setup

# Function to generate a list of random strings
def generate_random_strings(n=100000, length=64):
    return np.array([''.join(random.choices(string.ascii_letters + string.digits, k=length)).encode('utf-8') for _ in range(n)], dtype=np.bytes_)

# CUDA Kernel for SHA-256 Brute Force
@cuda.jit
def gpu_sha256(random_strings, target_hash, result_flag):
    idx = cuda.grid(1)

    if idx < random_strings.shape[0]:
        # Convert byte array to a string
        input_bytes = random_strings[idx]
        
        # Compute SHA-256 Hash
        hashed = hashlib.sha256(input_bytes).hexdigest()

        # If hash matches, set flag
        if hashed == target_hash:
            result_flag[0] = 1  # Signal that a match is found

# Function to process SHA-256 on a single GPU
def crack_with_gpu(gpu_id, target_hash, batch_size=100000):
    cuda.select_device(gpu_id)  # Assign GPU
    random_strings = generate_random_strings(batch_size)

    # Allocate device memory
    d_random_strings = cuda.to_device(random_strings)
    d_result_flag = cuda.to_device(np.array([0], dtype=np.int32))  # Flag for result

    # Launch Kernel
    threadsperblock = 256
    blockspergrid = (batch_size + (threadsperblock - 1)) // threadsperblock
    gpu_sha256[blockspergrid, threadsperblock](d_random_strings, target_hash, d_result_flag)

    # Copy result back
    result_flag = d_result_flag.copy_to_host()[0]

    if result_flag == 1:
        print(f"[GPU {gpu_id}] Found matching hash!")
    else:
        print(f"[GPU {gpu_id}] No match found.")

# Main function to run on multiple GPUs
def main(target_hash, num_gpus=NUM_GPUS):
    start_time = time.time()
    processes = []

    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(target=crack_with_gpu, args=(gpu_id, target_hash))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"Cracking completed in {time.time() - start_time:.2f} seconds.")

# Example target hash (replace with actual hash)
target_sha256 = "40c45198f179492a4008d19f4e67f7260ba728e9963e3af00d13eb46337ee1dc"

# Run with multiple GPUs
main(target_sha256, num_gpus=NUM_GPUS)

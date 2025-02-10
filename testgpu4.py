import hashlib
import random
import string
import time
import multiprocessing
import numpy as np
from numba import cuda

# Target SHA-256 hash (Replace with the actual target hash)
TARGET_HASH = "40c45198f179492a4008d19f4e67f7260ba728e9963e3af00d13eb46337ee1dc"

# Number of GPUs available (Change based on your setup)
NUM_GPUS = 4  

# Function to generate a random hexadecimal string (64 characters)
def generate_random_hex_string(length=64):
    return ''.join(random.choices("0123456789abcdef", k=length))

# GPU Kernel for iterative SHA-256 hashing
@cuda.jit
def gpu_sha256(target_hash, found_flag, output_string):
    idx = cuda.grid(1)
    
    if found_flag[0]:  # Stop if another GPU found the hash
        return

    # Generate a unique random hex string per GPU thread
    charset = "0123456789abcdef"
    rand_string = "".join(charset[idx % 16] for _ in range(64))  # Generate hex string

    # Hash it iteratively
    current_hash = hashlib.sha256(rand_string.encode()).hexdigest()

    while current_hash != target_hash:
        if found_flag[0]:  # Stop if another GPU found the hash
            return
        current_hash = hashlib.sha256(current_hash.encode()).hexdigest()

    # If found, store the matching string
    found_flag[0] = 1  # Set flag
    for i in range(len(rand_string)):
        output_string[i] = ord(rand_string[i])  # Copy string to output

# Function to run SHA-256 cracking on a single GPU
def crack_with_gpu(gpu_id, target_hash):
    cuda.select_device(gpu_id)  # Assign task to specific GPU

    found_flag = np.array([0], dtype=np.int32)
    output_string = np.zeros(64, dtype=np.uint8)  # Store found string

    d_found_flag = cuda.to_device(found_flag)
    d_output_string = cuda.to_device(output_string)

    threadsperblock = 256
    blockspergrid = 32

    print(f"[GPU {gpu_id}] Starting...")

    gpu_sha256[blockspergrid, threadsperblock](target_hash, d_found_flag, d_output_string)

    # Copy result back
    found_flag = d_found_flag.copy_to_host()[0]
    output_string = "".join(chr(c) for c in d_output_string.copy_to_host() if c != 0)

    if found_flag == 1:
        print(f"[GPU {gpu_id}] Found matching string: {output_string}")

# Main function to distribute work across multiple GPUs
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

# Run the script with multiple GPUs
main(TARGET_HASH, num_gpus=NUM_GPUS)
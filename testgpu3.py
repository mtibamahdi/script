import hashlib
import string
import time
import multiprocessing
from numba import cuda
import numpy as np

# SHA-256 hashing function
def sha256_hash(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()

# CUDA Kernel for SHA-256 Brute Force
@cuda.jit
def gpu_sha256(target_hash, results):
    idx = cuda.grid(1)

    while True:
        # Generate a random 64-character string (manual randomization)
        rand_string = ''
        for _ in range(64):
            rand_char = chr(97 + int(cuda.local.array(1, dtype=np.uint8)[0] % 26))  # Generate a lowercase letter
            rand_string += rand_char

        # Compute SHA-256 hash
        current_hash = hashlib.sha256(rand_string.encode()).hexdigest()

        if current_hash == target_hash:
            results[idx] = 1  # Mark found match
            break  # Stop the loop when found

# Function to run brute-force hashing on a GPU
def crack_with_gpu(gpu_id, target_hash):
    cuda.select_device(gpu_id)  # Assign task to specific GPU
    threads_per_block = 256
    blocks_per_grid = 32
    total_threads = threads_per_block * blocks_per_grid

    # Allocate GPU memory for results
    d_results = cuda.device_array(total_threads, dtype=np.int32)

    # Launch GPU kernel
    gpu_sha256[blocks_per_grid, threads_per_block](target_hash, d_results)

    # Copy results back to CPU
    results = d_results.copy_to_host()
    if any(results):
        print(f"[GPU {gpu_id}] Found matching hash!")

# Main function to run across multiple GPUs
def main(target_hash, num_gpus=2):
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
main(target_sha256, num_gpus=2)  # Adjust num_gpus based on available hardware

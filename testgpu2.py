import hashlib
import string
import time
import multiprocessing
from numba import cuda
import numpy as np
import numba

# SHA-256 hashing function
def sha256_hash(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()

# GPU Kernel for SHA-256 Hashing
@cuda.jit
def gpu_sha256(target_hash, results, rand_states):
    idx = cuda.grid(1)
    state = rand_states[idx]

    while True:
        # Generate a random string using CUDA-compatible random number generation
        rand_string = ''.join(chr(97 + int(cuda.random.xoroshiro128p_uniform_float32(state) * 26)) for _ in range(64))
        
        # Compute hash
        current_hash = hashlib.sha256(rand_string.encode()).hexdigest()

        if current_hash == target_hash:
            results[idx] = 1  # Indicate a found match
            break  # Exit the loop when found

# Function to initialize GPU random states
@cuda.jit
def init_rand_states(states, seed):
    idx = cuda.grid(1)
    states[idx] = numba.cuda.random.create_xoroshiro128p_states(1, seed + idx)

# Function to run hashing on a single GPU
def crack_with_gpu(gpu_id, target_hash):
    cuda.select_device(gpu_id)  # Assign task to specific GPU
    threadsperblock = 256
    blockspergrid = 32
    total_threads = threadsperblock * blockspergrid

    # Allocate device memory
    d_results = cuda.device_array(total_threads, dtype=np.int32)
    d_rand_states = cuda.device_array(total_threads, dtype=numba.cuda.random.Xoroshiro128pState)

    # Initialize random states
    init_rand_states[blockspergrid, threadsperblock](d_rand_states, int(time.time()))

    # Launch Kernel
    gpu_sha256[blockspergrid, threadsperblock](target_hash, d_results, d_rand_states)

    # Copy results back to CPU
    results = d_results.copy_to_host()
    if any(results):
        print(f"[GPU {gpu_id}] Found matching hash!")

# Main function to run on multiple GPUs
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
main(target_sha256, num_gpus=2)  # Change num_gpus based on available GPUs

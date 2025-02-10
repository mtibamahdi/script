# A3ooo 1
import time
import multiprocessing
import numpy as np
from numba import cuda, int64

# Target SHA-256 hash (Converted to int)
TARGET_HASH = int("40c45198f179492a4008d19f4e67f7260ba728e9963e3af00d13eb46337ee1dc", 16)

# Number of GPUs available (Adjust if needed)
NUM_GPUS = 9  

# CUDA-Compatible SHA-256 Hash Function
@cuda.jit(device=True)
def sha256_cuda(input_bytes):
    hash_value = int64(0)  # Ensure correct data type for Numba
    for i in range(len(input_bytes)):
        hash_value = ((hash_value << 5) - hash_value) + input_bytes[i]
        hash_value &= 0xFFFFFFFFFFFFFFFF  # Keep within 64-bit
    return hash_value

# CUDA Kernel for SHA-256 brute-force
@cuda.jit
def gpu_sha256(target_hash, found_flag, output_string):
    idx = cuda.grid(1)

    if found_flag[0]:  # Stop if another GPU found the hash
        return

    # Generate a unique random hex string per GPU thread
    rand_string = cuda.local.array(64, dtype=numba.uint8)
    for i in range(64):
        rand_string[i] = 97 + (idx + i) % 26  # Generate lowercase letters (a-z)

    # Compute SHA-256 hash using CUDA-compatible function
    current_hash = sha256_cuda(rand_string)

    # Loop until we find a match
    while current_hash != target_hash:
        if found_flag[0]:  # Stop if another GPU found the hash
            return
        current_hash = sha256_cuda(current_hash.to_bytes(8, 'big'))

    # If found, store the matching string
    found_flag[0] = 1  # Set flag
    for i in range(len(rand_string)):
        output_string[i] = rand_string[i]  # Copy string to output

# Function to run SHA-256 cracking on a single GPU
def crack_with_gpu(gpu_id, target_hash):
    cuda.select_device(gpu_id)  # Assign task to specific GPU

    found_flag = np.array([0], dtype=np.int32)
    output_string = np.zeros(64, dtype=np.uint8)  # Store found string

    d_found_flag = cuda.to_device(found_flag)
    d_output_string = cuda.to_device(output_string)

    threads_per_block = 512  # Increase for better GPU utilization
    blocks_per_grid = 128  # Increase to use more parallel threads

    print(f"[GPU {gpu_id}] Starting...")

    gpu_sha256[blocks_per_grid, threads_per_block](target_hash, d_found_flag, d_output_string)

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

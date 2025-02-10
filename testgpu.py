import hashlib
import random
import string
import time
import multiprocessing
from numba import cuda

# Function to generate a random string
def generate_random_string(length=64):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# SHA-256 hashing function
def sha256_hash(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()

# GPU Kernel for SHA-256 Hashing
@cuda.jit
def gpu_sha256(target_hash, results):
    idx = cuda.grid(1)
    found = False
    
    while not found:
        # Generate a random string (Modify to ensure uniqueness per GPU)
        rand_string = ''.join(random.choices(string.ascii_letters + string.digits, k=64))
        current_hash = hashlib.sha256(rand_string.encode()).hexdigest()
        
        if current_hash == target_hash:
            results[idx] = current_hash
            found = True  # Stop the loop when found

# Function to run Hashing on a single GPU
def crack_with_gpu(gpu_id, target_hash):
    cuda.select_device(gpu_id)  # Assign task to specific GPU
    d_results = cuda.device_array(1, dtype='int32')

    # Launch Kernel
    threadsperblock = 256
    blockspergrid = 32
    gpu_sha256[blockspergrid, threadsperblock](target_hash, d_results)

    if d_results.copy_to_host()[0] == target_hash:
        print(f"[GPU {gpu_id}] Found matching hash!")

# Main function to run on multiple GPUs
def main(target_hash, num_gpus=90):
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

# Run with 90 GPUs
main(target_sha256, num_gpus=90)
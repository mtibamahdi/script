import hashlib
import numpy as np
import time
import multiprocessing
from numba import cuda

# Target SHA-256 hash to match
TARGET_HASH = "40c45198f179492a4008d19f4e67f7260ba728e9963e3af00d13eb46337ee1dc"

# Number of GPUs available (Adjust based on your system)
NUM_GPUS = 4  

@cuda.jit
def gpu_sha256(target_hash, found_flag, output_string):
    """CUDA kernel to generate random SHA-256 hashes and check if they match the target hash."""
    idx = cuda.grid(1)  # Unique thread index

    if found_flag[0]:  # If another thread found it, stop
        return

    # Generate a random 64-character hex string
    charset = b"0123456789abcdef"
    rand_string = cuda.local.array(64, dtype=np.uint8)
    
    for i in range(64):
        rand_string[i] = charset[(idx + i) % 16]  # Create a unique input for each thread

    # Convert to bytes
    input_data = bytes(rand_string)

    # Compute SHA-256 hash
    hashed_once = hashlib.sha256(input_data).hexdigest()

    # Compare with target hash
    if hashed_once == target_hash.decode():
        found_flag[0] = 1  # Signal that we found the hash
        for i in range(64):  # Copy found string to output
            output_string[i] = rand_string[i]

def crack_with_gpu(gpu_id, target_hash):
    """Function to run the brute-force SHA-256 cracking on a single GPU."""
    cuda.select_device(gpu_id)

    found_flag = np.array([0], dtype=np.int32)  # Flag to indicate if a match is found
    output_string = np.zeros(64, dtype=np.uint8)  # Store found preimage

    d_found_flag = cuda.to_device(found_flag)
    d_output_string = cuda.to_device(output_string)

    threadsperblock = 256
    blockspergrid = 32

    print(f"[GPU {gpu_id}] Starting...")

    gpu_sha256[blockspergrid, threadsperblock](target_hash.encode(), d_found_flag, d_output_string)

    # Copy results back to host
    found_flag = d_found_flag.copy_to_host()[0]
    output_string = "".join(chr(c) for c in d_output_string.copy_to_host() if c != 0)

    if found_flag == 1:
        print(f"[GPU {gpu_id}] Found matching preimage: {output_string}")

def main(target_hash, num_gpus=NUM_GPUS):
    """Main function to distribute work across multiple GPUs."""
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
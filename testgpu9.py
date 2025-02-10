import cupy as cp
import hashlib
import string
import random
from concurrent.futures import ThreadPoolExecutor

# Generate a batch of random strings on the GPU
def generate_random_strings(batch_size, length=64):
    chars = string.ascii_letters + string.digits
    return [''.join(random.choices(chars, k=length)) for _ in range(batch_size)]

# SHA-256 hashing function using CuPy
def sha256_gpu(input_strings):
    # Convert input strings to byte arrays
    input_bytes = cp.array([s.encode('utf-8') for s in input_strings], dtype=cp.uint8)

    # CuPy SHA-256 Kernel (vectorized hashing)
    sha256_kernel = cp.ElementwiseKernel(
        'raw uint8 input_bytes, int32 length',
        'uint8 output_hash[32]',
        '''
        unsigned char digest[32];
        sha256(input_bytes, length, digest);
        for (int i = 0; i < 32; i++) {
            output_hash[i] = digest[i];
        }
        ''',
        'sha256_gpu_kernel'
    )

    # Compute hashes on GPU
    output_hashes = cp.empty((len(input_strings), 32), dtype=cp.uint8)
    sha256_kernel(input_bytes, len(input_strings[0]), output_hashes)
    
    return output_hashes

# Convert hash string to GPU array
def hex_to_gpu_array(hex_str):
    return cp.array(list(bytes.fromhex(hex_str)), dtype=cp.uint8)

# Multi-GPU brute-force function
def find_hash_multi_gpu(target_hash, batch_size=10000):
    target_hash_gpu = hex_to_gpu_array(target_hash)  # Convert target hash to GPU array
    num_gpus = cp.cuda.runtime.getDeviceCount()  # Get available GPU count

    def worker(gpu_id):
        cp.cuda.Device(gpu_id).use()  # Set current GPU
        while True:
            batch = generate_random_strings(batch_size)  # Generate batch of random strings
            hash_results = sha256_gpu(batch)  # Compute hashes on GPU
            matches = cp.where(cp.all(hash_results == target_hash_gpu, axis=1))[0]  # Find match

            if matches.size > 0:
                return batch[matches[0]]  # Return the matching input string

    # Use ThreadPoolExecutor to run multiple GPUs in parallel
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        results = executor.map(worker, range(num_gpus))
        for result in results:
            if result:
                print(f"Found matching input: {result}")
                return result

# Example target SHA-256 hash
target_sha256 = "40c45198f179492a4008d19f4e67f7260ba728e9963e3af00d13eb46337ee1dc"

# Run the multi-GPU brute-force search
find_hash_multi_gpu(target_sha256)
import hashlib
import random
import string
import cupy as cp
import numpy as np
import multiprocessing

# Number of GPUs available
NUM_GPUS = cp.cuda.runtime.getDeviceCount()  # Automatically detect GPU count

# Function to generate a random string
def generate_random_string(length=64):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Function to hash the string using SHA-256
def sha256_hash(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()

# Convert the hash to a GPU array of integers
def sha256_gpu(input_string):
    hash_result = hashlib.sha256(input_string.encode()).hexdigest()
    byte_array = bytes.fromhex(hash_result)
    return cp.asarray(np.frombuffer(byte_array, dtype=np.uint8))  # Convert to uint8

# GPU worker function
def gpu_worker(gpu_id, target_hash):
    cp.cuda.Device(gpu_id).use()  # Assign specific GPU
    print(f"[GPU {gpu_id}] Starting search...")

    # Convert the target hash to a GPU array
    target_hash_bytes = bytes.fromhex(target_hash)
    target_hash_gpu = cp.asarray(np.frombuffer(target_hash_bytes, dtype=np.uint8))

    attempt = 0
    while True:
        # Generate a new random string per GPU
        random_string = generate_random_string()
        current_hash = sha256_hash(random_string)
        current_hash_gpu = sha256_gpu(current_hash)

        # Check if we found a match
        if cp.array_equal(current_hash_gpu[:6], target_hash_gpu[:6]):  # Compare first 6 bytes
            print(f"✅ [GPU {gpu_id}] Found matching hash: {current_hash} in {attempt} attempts!")
            return current_hash  # Return the final hash that matches the target

        attempt += 1
        if attempt % 1000 == 0:
            print(f"[GPU {gpu_id}] Attempts: {attempt}, Current hash: {current_hash}")

# Main function to launch multiple GPUs
def main(target_hash, num_gpus=NUM_GPUS):
    processes = []
    
    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(target=gpu_worker, args=(gpu_id, target_hash))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("🎉 Brute force search completed.")

# Example target hash (replace with the actual target hash you're looking for)
target_sha256 = "40c45198f179492a4008d19f4e67f7260ba728e9963e3af00d13eb46337ee1dc"

# Run the search across multiple GPUs
if __name__ == "__main__":
    main(target_sha256)

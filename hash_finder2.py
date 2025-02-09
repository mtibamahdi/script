import hashlib
import random
import string
import cupy as cp

# Function to generate a random string
def generate_random_string(length=64):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Function to hash the string using SHA-256
def sha256_hash(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()

# Convert the hash to a GPU array of bytes
def sha256_gpu(input_string):
    hash_result = hashlib.sha256(input_string.encode()).hexdigest()
    # Convert hash string to a byte array, and then into a CuPy array
    byte_array = bytes.fromhex(hash_result)
    return cp.asarray(byte_array)

# Function to repeatedly hash using GPU until target hash is found
def find_matching_hash_gpu(target_hash):
    # Step 1: Generate a random string and hash it
    random_string = generate_random_string()
    current_hash = sha256_hash(random_string)
    print(f"Initial random hash: {current_hash}")

    # Convert the target hash to a GPU array of bytes
    target_hash_bytes = bytes.fromhex(target_hash)
    target_hash_gpu = cp.asarray(target_hash_bytes)

    # Convert the initial hash to a GPU array
    current_hash_gpu = sha256_gpu(current_hash)

    # Step 2: Repeatedly hash using GPU until we find the target
    while not cp.array_equal(current_hash_gpu, target_hash_gpu):
        current_hash = sha256_hash(current_hash)  # Hash the current hash string
        current_hash_gpu = sha256_gpu(current_hash)  # Convert to GPU array

    print(f"Found matching hash: {current_hash}")
    return current_hash  # Return the final hash that matches the target

# Example target hash (replace with the target SHA-256 hash you're looking for)
target_sha256 = "40c45198f179492a4008d19f4e67f7260ba728e9963e3af00d13eb46337ee1dc"  # Target hash

# Run the search
find_matching_hash_gpu(target_sha256)
import hashlib
import random
import string

# Function to generate a random string
def generate_random_string(length=64):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Function to hash the string using SHA-256
def sha256_hash(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()

# Function to repeatedly hash the initial random SHA-256 hash until target hash is found
def find_matching_hash(target_hash):
    # Step 1: Generate a random string and hash it
    random_string = generate_random_string()
    current_hash = sha256_hash(random_string)
    print(f"Initial random hash: {current_hash}")
    
    # Step 2: Repeatedly hash the current hash until we find the target
    while current_hash != target_hash:
        current_hash = sha256_hash(current_hash)
        print(f"Current hash: {current_hash}")  # Optional: Print each hash step for debugging

    print(f"Found matching hash: {current_hash}")
    return current_hash  # Return the final hash that matches the target

# Example target hash (replace with the target SHA-256 hash you're looking for)
target_sha256 = "40c45198f179492a4008d19f4e67f7260ba728e9963e3af00d13eb46337ee1dc"  # Target hash

# Run the search
find_matching_hash(target_sha256)
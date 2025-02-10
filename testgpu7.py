from numba import cuda
import numpy as np

# Simple CUDA kernel to add 1 to each element in an array
@cuda.jit
def gpu_test(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] += 1  # Increment each element by 1

# Function to test CUDA execution
def test_gpu():
    print("🔄 Testing CUDA functionality...")

    # Create a NumPy array on CPU
    size = 10  # Small test array
    host_array = np.arange(size, dtype=np.int32)

    # Allocate memory on GPU
    device_array = cuda.to_device(host_array)

    # Define GPU execution configuration
    threads_per_block = 32
    blocks_per_grid = (size + (threads_per_block - 1)) // threads_per_block

    # Execute kernel
    gpu_test[blocks_per_grid, threads_per_block](device_array)

    # Copy result back to CPU
    result_array = device_array.copy_to_host()

    # Print results
    print("✅ GPU computation completed!")
    print("Original Array:", np.arange(size, dtype=np.int32))
    print("Modified Array (After GPU Processing):", result_array)

# Run the test function
if __name__ == "__main__":
    test_gpu()

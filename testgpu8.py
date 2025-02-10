from numba import cuda
import numpy as np

# Simple CUDA kernel to add 1 to each element in an array
@cuda.jit
def gpu_test(arr):
    idx = cuda.grid(1)  # Get thread index
    if idx < arr.size:
        arr[idx] += 1  # Increment each element by 1

# Function to test CUDA execution with optimized grid size
def test_gpu():
    print("🔄 Testing CUDA functionality...")

    # Create a NumPy array on CPU
    size = 1000000  # Large array to utilize GPU fully
    host_array = np.arange(size, dtype=np.int32)

    # Allocate memory on GPU
    device_array = cuda.to_device(host_array)

    # Define GPU execution configuration
    threads_per_block = 256
    blocks_per_grid = (size + (threads_per_block - 1)) // threads_per_block  # Optimized grid size

    # Execute kernel
    gpu_test[blocks_per_grid, threads_per_block](device_array)

    # Copy result back to CPU
    result_array = device_array.copy_to_host()

    # Print results (only first 10 values)
    print("✅ GPU computation completed!")
    print("Original Array (First 10):", np.arange(10, dtype=np.int32))
    print("Modified Array (First 10, After GPU Processing):", result_array[:10])

# Run the test function
if __name__ == "__main__":
    test_gpu()

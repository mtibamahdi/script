import subprocess

def run_hashcat(target_hash):
    # Hashcat command to use all 9 GPUs and brute force a 64-character hex string
    command = [
        "hashcat", "-m", "1400",  # 1400 = SHA-256
        "-a", "3",  # Attack mode 3 = brute force
        "--backend-devices=all",  # Use all GPUs
        "--status",  # Show progress
        "--status-timer=10",  # Update status every 10 seconds
        target_hash, "?h" * 64  # 64-character hex brute force
    ]

    print(f"🔥 Running Hashcat on 9 GPUs for target hash: {target_hash}")
    subprocess.run(command)

# Example target SHA-256 hash
target_sha256 = "40c45198f179492a4008d19f4e67f7260ba728e9963e3af00d13eb46337ee1dc"

run_hashcat(target_sha256)
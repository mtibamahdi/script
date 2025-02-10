import subprocess

def run_hashcat(target_hash):
    # Hashcat command to use all 9 GPUs and brute force a 64-character hex string
    command = [
    "hashcat", "-m", "1400",
    "-a", "3",
    "--status",
    "--status-timer=10",
    target_hash, "?h" * 64
]


    print(f"🔥 Running Hashcat on 9 GPUs for target hash: {target_hash}")
    subprocess.run(command)

# Example target SHA-256 hash
target_sha256 = "40c45198f179492a4008d19f4e67f7260ba728e9963e3af00d13eb46337ee1dc"

run_hashcat(target_sha256)
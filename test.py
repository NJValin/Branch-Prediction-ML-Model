from rich.progress import Progress
import sys
from tqdm import tqdm
import time

with Progress() as progress:
    task = progress.add_task("[cyan]Processing...", total=100)
    for _ in range(100):
        progress.update(task, advance=1)
        time.sleep(0.05)


for _ in tqdm(range(100), desc="Processing", unit="step"):
    time.sleep(0.05)  # Simulate work


def pseudo_progress_bar(duration=60, length=40):
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        percent = (elapsed / duration) * 100
        filled_length = int(length * percent // 100)
        bar = "█" * filled_length + "-" * (length - filled_length)
        sys.stdout.write(f"\r|{bar}| {percent:.2f}%")
        sys.stdout.flush()
        time.sleep(0.5)  # Adjust refresh rate
    print("\nDone!")

# Simulating a 10-second process
pseudo_progress_bar(duration=10)


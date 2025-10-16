import subprocess
import multiprocessing
from tqdm import tqdm

# Define command parameters
original_data_path = "/home/liuzhenghao/nuScenes_Carla_multi_drone/CoSwarm"
create_data_save_path = "/home/liuzhenghao/dataset/CoSwarm-det-test"
from_agent = "0"
to_agent = "8"
# Remember to change the `max_num_agent` in "from_file_multisweep_warp2com_sample_data"
num_processes = 25
assert 100 % num_processes == 0
num_scene_per_processes = 100 // num_processes


# Define the function to be executed by each process
def process_scene(scene):
    command = [
        "python",
        "create_data_det.py",
            "--root", original_data_path,
            "--savepath", create_data_save_path,
            "--from_agent", from_agent,
            "--to_agent", to_agent,
            "--scenes",
    ]
    command.extend(list(map(str, scene)))
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing scene {scene}: {e}")
        return False


if __name__ == "__main__":
    # Create a process pool using the number of CPU cores
    pool = multiprocessing.Pool(processes=num_processes)
    # Generate scene list: symmetric to ensure each process handles one 5-agent and one 8-agent scene
    scenes = [[i, i+25, i+50, i+75] for i in range(num_processes)]
    # Use tqdm to display progress bar
    results = list(tqdm(pool.imap(process_scene, scenes), total=len(scenes)))
    # Close the process pool
    pool.close()
    # Wait for all processes to complete
    pool.join()
    # Count the number of successfully processed scenes
    success_count = sum(results)
    print(f"All scene processing completed. Successfully processed {success_count} scenes, failed {len(scenes) - success_count} scenes.")
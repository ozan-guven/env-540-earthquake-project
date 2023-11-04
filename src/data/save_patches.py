import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import concurrent.futures

Image.MAX_IMAGE_PIXELS = 303038464

DATA_PATH = '../../data/'
MAXAR_PATH = DATA_PATH + 'maxar/'
MAXAR_POST_PATH = MAXAR_PATH + 'post/'
MAXAR_PRE_PATH = MAXAR_PATH + 'pre/'
MAXAR_PATCHES_PATH = DATA_PATH + 'maxar_patches/'
MAXAR_POST_PATCHES_PATH = MAXAR_PATCHES_PATH + 'post/'
MAXAR_PRE_PATCHES_PATH = MAXAR_PATCHES_PATH + 'pre/'

def get_image_files(directory):
    image_files = []
    for query_key_dir in os.listdir(directory):
        query_key_dir = f"{directory}/{query_key_dir}"
        for file in os.listdir(query_key_dir):
            image_files.append(f"{query_key_dir}/{file}")

    return image_files

def get_image_file_patches(
          image_file, 
          image_size=17_408, 
          patch_size=1024,
          max_black_ratio=0.8,
          black_threshold = 10, 
          snow_threshold = 500,
          variance_threshold=0.04 * 255 * 255,
          ):
    if image_size % patch_size != 0:
            raise ValueError('❌ Image size must be divisible by patch size')
    
    # Get image
    image = Image.open(image_file)
    if image.size != (image_size, image_size):
        raise ValueError(f'❌ Image size must be {image_size}x{image_size}')
    
    # Get patches
    patches = []
    for y in range(0, image_size, patch_size):
        for x in range(0, image_size, patch_size):
            patch = image.crop((x, y, x+patch_size, y+patch_size))
            patches.append(patch)

    # Filter out patches with too many black pixels or too much snow
    filtered_patches = []
    filtered_patch_ids = []
    for i, patch in enumerate(patches):
        sum_patch = np.sum(np.array(patch), axis=2)
        mean_patch = np.mean(np.array(patch), axis=2)

        # Black pixels
        n_total_pixels = patch_size * patch_size
        n_black_pixels = np.sum(sum_patch < black_threshold)
        black_ratio = n_black_pixels / n_total_pixels
        if black_ratio > max_black_ratio:
            continue

        # Snow
        non_black_pixels = sum_patch[sum_patch > black_threshold]
        if np.mean(non_black_pixels) > snow_threshold:
            continue

        # Variance
        non_black_pixels = mean_patch[sum_patch > black_threshold]
        variance = np.var(non_black_pixels)
        if variance < variance_threshold:
            continue

        filtered_patches.append(patch)
        filtered_patch_ids.append(i)

    return filtered_patches, filtered_patch_ids

def save_image_file_patches(
        image_file,
        save_dir,
        image_size=17_408,
        patch_size=1024,
        max_black_ratio=0.8,
        ):
    patches, patch_ids = get_image_file_patches(
        image_file,
        image_size=image_size,
        patch_size=patch_size,
        max_black_ratio=max_black_ratio,
        )
    
    # Create save directory if necessary
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save patches
    for patch, patch_id in zip(patches, patch_ids):
        patch_file = f"{save_dir}/{image_file.split('/')[-2]}_{image_file.split('/')[-1].split('.')[0]}_{patch_id}.png"
        if not os.path.exists(patch_file):
            patch.save(patch_file)

def query_cpu_count():
    cpu_count = input('How many CPUs to use? (default: all) ')
    if cpu_count == '':
        cpu_count = os.cpu_count()
    else:
        cpu_count = int(cpu_count)

    return cpu_count

if __name__ == '__main__':
    num_workers = query_cpu_count()

    # Save post patches
    post_image_files = get_image_files(MAXAR_POST_PATH)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        progress = tqdm(total=len(post_image_files), desc='🔄 Patching post images')
        futures = [executor.submit(save_image_file_patches, post_image_file, MAXAR_POST_PATCHES_PATH) for post_image_file in post_image_files]
        for future in concurrent.futures.as_completed(futures):
            progress.update(1)
        progress.close()

    # Save pre patches
    pre_image_files = get_image_files(MAXAR_PRE_PATH)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        progress = tqdm(total=len(pre_image_files), desc='🔄 Patching pre images')
        futures = [executor.submit(save_image_file_patches, pre_image_file, MAXAR_PRE_PATCHES_PATH) for pre_image_file in pre_image_files]
        for future in concurrent.futures.as_completed(futures):
            progress.update(1)
        progress.close()



"""
    This file contains a script to filter maxar patches. It creates a new 'maxar_filtered_patches' folder
    containing all patches from the 'maxar_patches' folder. Files are saved in two folders: 'post' and 'pre',
    each containing patches from the post and pre disaster images respectively. Any pre-image should have its
    corresponding post-image in the 'post' folder. The script uses ThreadPoolExecutor to filter patches in
    parallel.
"""

import os
import shutil
from tqdm import tqdm
from typing import List

DATA_PATH = '../../data/'
MAXAR_PATCHES_PATH = DATA_PATH + 'maxar_patches/'
MAXAR_POST_PATCHES_PATH = MAXAR_PATCHES_PATH + 'post/'
MAXAR_PRE_PATCHES_PATH = MAXAR_PATCHES_PATH + 'pre/'
MAXAR_FILTERED_PATCHES_PATH = DATA_PATH + 'maxar_filtered_patches/'
MAXAR_PRE_FILTERED_PATCHES_PATH = MAXAR_FILTERED_PATCHES_PATH + 'pre/'
MAXAR_POST_FILTERED_PATCHES_PATH = MAXAR_FILTERED_PATCHES_PATH + 'post/'

def _create_data_folders() -> None:
    """Create pre and post earthquake folders for data.
    """
    print('▶️ Creating data folders...')

    os.makedirs(MAXAR_FILTERED_PATCHES_PATH, exist_ok=True)
    os.makedirs(MAXAR_PRE_FILTERED_PATCHES_PATH, exist_ok=True)
    os.makedirs(MAXAR_POST_FILTERED_PATCHES_PATH, exist_ok=True)

def _get_short_patch_file_name(file: str) -> str:
    """Get short file name from patch file name.

    Args:
        file (str): Patch file name.

    Returns:
        short_file (str): Short file name.
    """
    short_file = file.split('/')[-1]
    short_file = short_file.split('_')[0] + '_' + short_file.split('_')[3]

    return short_file

def _get_patch_files() -> List[str]:
    # Get all patch files
    pre_patch_files = [f"{MAXAR_PRE_PATCHES_PATH}{file}" for file in os.listdir(MAXAR_PRE_PATCHES_PATH)]
    post_patch_files = [f"{MAXAR_POST_PATCHES_PATH}{file}" for file in os.listdir(MAXAR_POST_PATCHES_PATH)]
    print(len(pre_patch_files), len(post_patch_files))

    # Get short file names
    short_pre_patch_files = set(_get_short_patch_file_name(file) for file in pre_patch_files)
    short_post_patch_files = set(_get_short_patch_file_name(file) for file in post_patch_files)

    # Get set of corresponding short file names
    same_patch_files = short_pre_patch_files.intersection(short_post_patch_files)

    # Get corresponding patch files
    pre_patch_files = [file for file in pre_patch_files if _get_short_patch_file_name(file) in same_patch_files]
    post_patch_files = [file for file in post_patch_files if _get_short_patch_file_name(file) in same_patch_files]

    return pre_patch_files, post_patch_files

def _save_patch_files(pre_patch_files, post_patch_files) -> None:
    """Save patch files.

    Args:
        pre_patch_files (List[str]): List of pre patch file paths.
        post_patch_files (List[str]): List of post patch file paths.
    """
    print('▶️ Saving patch files...')

    # Save pre patch files
    for file in tqdm(pre_patch_files, desc='🔄 Saving pre patch files'):
        shutil.copy(file, MAXAR_PRE_FILTERED_PATCHES_PATH)

    # Save post patch files
    for file in tqdm(post_patch_files, desc='🔄 Saving post patch files'):
        shutil.copy(file, MAXAR_POST_FILTERED_PATCHES_PATH)

if __name__ == '__main__':
    _create_data_folders()
    pre_patch_files, post_patch_files = _get_patch_files()
    _save_patch_files(pre_patch_files, post_patch_files)
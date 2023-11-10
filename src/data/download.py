"""
    This file contains a script to download data from Maxar's Open Data Program. It creates a new 'maxar' folder
    containing all data from the links.txt file. The data is sorted into pre and post earthquake folders based on
    the date of the image. The script uses ThreadPoolExecutor to download files in parallel.
"""

import os
from tqdm import tqdm
from typing import List
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_PATH = '../../data/'
LINKS_PATH = DATA_PATH + 'links.txt'
MAXAR_PATH = DATA_PATH + 'maxar/'
MAXAR_PRE_PATH = MAXAR_PATH + 'pre/'
MAXAR_POST_PATH = MAXAR_PATH + 'post/'

EARTHQUAKE_DATE = '2023-02-06'

def get_urls() -> List[str]:
    """Get urls from text file.

    Returns:
        urls (list): List of urls.
    """
    with open(LINKS_PATH, 'r') as f:
        urls = f.readlines()
    return urls

def create_data_folders(urls: str) -> None:
    """Create pre and post earthquake folders for data.

    Args:
        urls (list): List of urls.
    """
    print('▶️ Creating data folders...')

    os.makedirs(MAXAR_PATH, exist_ok=True)
    os.makedirs(MAXAR_PRE_PATH, exist_ok=True)
    os.makedirs(MAXAR_POST_PATH, exist_ok=True)

    for url in urls:
        url_string = url.split('/')[-3:]
        date = url_string[1]
        file_path = MAXAR_PRE_PATH if date < EARTHQUAKE_DATE else MAXAR_POST_PATH
        file_path += url_string[0] + '/'

        os.makedirs(file_path, exist_ok=True)

def download_file(url: str) -> None:
    """Download a file from a url and save to the specified path.

    Args:
        url (str): URL of the file to download.
        maxar_path (str): Path to save files.
        earthquake_date (str): Date to compare for pre/post folder sorting.
    """
    url_string = url.split('/')[-3:]
    date = url_string[1]
    file_path = MAXAR_PRE_PATH if date < EARTHQUAKE_DATE else MAXAR_POST_PATH
    file_path += url_string[0] + '/'
    file_name = '_'.join(url_string[-2:]).strip()
    file_path_with_name = os.path.join(file_path, file_name)

    if not os.path.exists(file_path_with_name):
        urlretrieve(url, file_path_with_name)

def download_files(urls: List[str]) -> None:
    """Download files from urls in parallel.

    Args:
        urls (List[str]): List of URLs to download.
        maxar_path (str): Path to save files.
        earthquake_date (str): Date to compare for pre/post folder sorting.
    """
    print ('▶️ Downloading files...')

    # Use ThreadPoolExecutor to download files in parallel
    with ThreadPoolExecutor() as executor:
        # Create a list to hold the futures
        future_to_url = {executor.submit(download_file, url): url for url in urls}
        
        # Iterate over the futures as they complete
        for future in tqdm(as_completed(future_to_url), total=len(urls), desc="🔄 Downloading files.", unit="file"):
            url = future_to_url[future]
            try:
                future.result()
            except Exception as exc:
                print(f'❌ {url} generated an exception: {exc}')

if __name__ == '__main__':
    urls = get_urls()
    create_data_folders(urls)
    download_files(urls)

import os
from tqdm import tqdm
from typing import List
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_PATH = '../../data/'
LINKS_PATH = DATA_PATH + 'links.txt'
MAXAR_PATH = DATA_PATH + 'maxar/'

EARTHQUAKE_DATE = '2023-02-06'

def get_urls(links_path: str) -> List[str]:
    """Get urls from text file.

    Args:
        links_path (str): Path to text file containing urls.

    Returns:
        urls (list): List of urls.
    """
    with open(links_path, 'r') as f:
        urls = f.readlines()
    return urls

def download_file(url, maxar_path, earthquake_date):
    """Download a file from a url and save to the specified path.

    Args:
        url (str): URL of the file to download.
        maxar_path (str): Path to save files.
        earthquake_date (str or datetime): Date to compare for pre/post folder sorting.
    """
    url_string = url.split('/')[-3:]
    date = url_string[1]
    file_path = maxar_path
    if date < earthquake_date:
        file_path += 'pre/'
    elif date >= earthquake_date:
        file_path += 'post/'

    file_path += url_string[0] + '/'
    file_name = '_'.join(url_string[-2:]).strip()

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path_with_name = os.path.join(file_path, file_name)

    if not os.path.exists(file_path_with_name):
        urlretrieve(url, file_path_with_name)

def download_files(urls, maxar_path, earthquake_date):
    """Download files from urls in parallel.

    Args:
        urls (list): List of URLs to download.
        maxar_path (str): Path to save files.
        earthquake_date (str or datetime): Date to compare for pre/post folder sorting.
    """
    # Use ThreadPoolExecutor to download files in parallel
    with ThreadPoolExecutor() as executor:
        # Create a list to hold the futures
        future_to_url = {executor.submit(download_file, url, maxar_path, earthquake_date): url for url in urls}
        
        # Iterate over the futures as they complete
        for future in tqdm(as_completed(future_to_url), total=len(urls), desc="Downloading", unit="file"):
            url = future_to_url[future]
            try:
                # Result is None, since the download_file function does not return anything
                future.result()
            except Exception as exc:
                print(f'{url} generated an exception: {exc}')

if __name__ == '__main__':
    urls = get_urls(LINKS_PATH)
    download_files(urls, MAXAR_PATH, EARTHQUAKE_DATE)

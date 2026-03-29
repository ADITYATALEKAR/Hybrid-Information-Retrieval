import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import re

# Extract File ID from the Google Drive URL
def extract_file_id(url):
    print(f"Extracting file ID from URL: {url}")
    match = re.search(r'd/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        print(f"Extracted File ID: {file_id}")
        return file_id
    else:
        raise ValueError("File ID could not be extracted from the URL")

# Download the file from Google Drive using the file ID
def download_file(service, file_id, file_name, download_folder):
    try:
        # Check if file already exists in the folder
        file_path = os.path.join(download_folder, file_name)
        if os.path.exists(file_path):
            print(f"File {file_name} already exists. Skipping download.")
            return

        # Download the file if it doesn't exist
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {file_name} {int(status.progress() * 100)}%.")
    except Exception as e:
        print(f"Error downloading file: {e}")

# Main function to download files from a list of Google Drive URLs
def download_files_from_urls(file_urls, download_folder):
    # Build the Google Drive API service with your API key (no authentication required)
    service = build('drive', 'v3', developerKey='AIzaSyCCjokRnFEaPpnhxO1BlAw0q2O1IQdNSSA')  # Use your API key here

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    for file_url in file_urls:
        # Extract file ID from the URL
        file_id = extract_file_id(file_url)

        try:
            # Get file metadata (to retrieve file name)
            file_metadata = service.files().get(fileId=file_id).execute()
            file_name = file_metadata['name']
            print(f"File Metadata: {file_metadata}")

            # Download the file
            print(f"Downloading: {file_name}")
            download_file(service, file_id, file_name, download_folder)
            print(f"Downloaded {file_name} to {download_folder}.")

        except Exception as e:
            print(f"Error retrieving file metadata or downloading file: {e}")

if __name__ == '__main__':
    # List of file URLs to download
    file_urls = [
        'https://drive.google.com/file/d/1Jw9rtdG6rjZhTTFmcO6o2TaOmhpGK96z/view?usp=sharing',
        'https://drive.google.com/file/d/1CDaJfecagT5ZF9PdhxohhTuDFJQnbYCn/view?usp=sharing',# Add more URLs here
        'https://drive.google.com/file/d/1eFvYyW_OAstK4Y9LZFRkLTV_-55jMHwt/view?usp=sharing'
    ]
    download_folder = './Model'  # Specify the folder to download the files into

    # Download files from the provided Google Drive URLs
    download_files_from_urls(file_urls, download_folder)

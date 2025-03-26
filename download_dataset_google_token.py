import os
import gdown
import time
import requests
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(InsecureRequestWarning)

def download_dataset():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Google Drive file ID from the URL
    file_id = '1N9gioWnkb3ZZytmT3Nzx4VjXjHxLsVB9'
    
    # Output path
    output_path = 'data/Human_Attribute_Pretrain.tar.gz'
    
    # Prompt user for the Google Drive API token
    print("Please enter your Google Drive API token.")
    print("You can get a token by:")
    print("1. Going to https://developers.google.com/oauthplayground/")
    print("2. Select Drive API v3 in the list on the right")
    print("3. Click 'Authorize APIs'")
    print("4. Click 'Exchange authorization code for tokens'")
    print("5. Copy the 'Access token' value")
    token = input("Enter your Google Drive API token: ")
    
    if not token:
        print("No token provided. Falling back to gdown method.")
        use_token_method = False
    else:
        use_token_method = True
    
    print(f"Downloading dataset...")
    print("This might take a while depending on your internet connection...")
    
    # Method 1: Using direct download with token
    if use_token_method:
        try:
            # Direct download URL
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
            
            # Headers with authorization
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": "Mozilla/5.0"
            }
            
            print(f"Starting direct download using Google API...")
            
            # Stream download to handle large files
            with requests.get(url, headers=headers, stream=True, verify=False) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                print(f"Total file size: {total_size / (1024*1024*1024):.2f} GB")
                
                with open(output_path, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192*1024):  # 8MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            # Print progress
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rDownloaded: {downloaded / (1024*1024*1024):.2f} GB ({percent:.1f}%)", end="")
                
                print(f"\nDownload completed! File saved to: {output_path}")
                return
        except Exception as e:
            print(f"Direct download failed: {str(e)}")
            print("Trying alternative method with gdown...")
    
    # Method 2: Fall back to gdown if direct download fails or no token provided
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} using gdown...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False, verify=False)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024*1024:  # > 1MB
                print(f"\nDownload completed! File saved to: {output_path}")
                print(f"File size: {os.path.getsize(output_path) / (1024*1024*1024):.2f} GB")
                break
            else:
                raise Exception("Download failed or file is too small")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Error downloading the file after {max_retries} attempts: {str(e)}")
                print("Please try downloading manually or check your internet connection.")

if __name__ == "__main__":
    download_dataset()
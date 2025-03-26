import os
import gdown

def download_dataset():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Google Drive file ID from the URL

    # Tiktok_finetuning.tar.gz (7.2G)
    # file_id = '1_b4naNB1QozGL-tKyHwSSYzTw8RIh5z3'

    # Human_Attribute_Pretrain.tar.gz (52G) 
    file_id = '1N9gioWnkb3ZZytmT3Nzx4VjXjHxLsVB9' 
    # Output path
    output_path = 'data/Human_Attribute_Pretrain.tar.gz'


    # Download the file
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Downloading dataset from {url}")
    print("This might take a while depending on your internet connection...")
    
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"\nDownload completed! File saved to: {output_path}")
    except Exception as e:
        print(f"Error downloading the file: {str(e)}")

if __name__ == "__main__":
    download_dataset() 
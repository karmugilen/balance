
import os
from PIL import Image
import glob

def resize_images(directory, size=(256, 256)):
    files = glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg"))
    print(f"Found {len(files)} images in {directory}")
    
    for f in files:
        try:
            img = Image.open(f)
            img = img.resize(size, Image.LANCZOS)
            img.save(f)
            print(f"Resized {f} to {size}")
        except Exception as e:
            print(f"Error resizing {f}: {e}")

if __name__ == "__main__":
    resize_images("my_images")

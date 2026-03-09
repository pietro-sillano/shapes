import os
import glob
from PIL import Image
from pathlib import Path
from collections import defaultdict
import re

def create_gifs_from_pngs(plot_folder="plot", output_folder="gifs"):
    """
    Create GIFs for each RPA from PNG files, sorting by deg parameter in ascending order.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Group PNG files by RPA
    rpa_images = defaultdict(list)
    
    # Find all PNG files
    png_files = glob.glob(os.path.join(plot_folder, "*.png"))
    
    # Extract RPA and deg from filenames and organize
    for png_file in png_files:
        filename = os.path.basename(png_file)
        # Adjust regex based on your actual filename pattern
        match = re.search(r"ms_rpa_(\d+\.?\d*)_phi_(\d+\.?\d*)", filename, re.IGNORECASE)
        if match:
            rpa = match.group(1)
            deg = float(match.group(2))
            rpa_images[rpa].append((deg, png_file))
    
    # Create GIF for each RPA
    for rpa in sorted(rpa_images.keys()):
        # Sort by deg in ascending order
        sorted_images = sorted(rpa_images[rpa], key=lambda x: x[0])
        
        # Open images
        images = []
        for deg, png_file in sorted_images:
            images.append(Image.open(png_file))
        
        if images:
            # Save as GIF
            output_path = os.path.join(output_folder, f"rpa_{rpa}.gif")
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=500,  # 500ms per frame
                loop=0
            )
            print(f"Created: {output_path}")

if __name__ == "__main__":
    create_gifs_from_pngs()
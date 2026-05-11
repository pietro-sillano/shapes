import os
import glob
from PIL import Image
from pathlib import Path
from collections import defaultdict
import re


def create_gifs_from_pngs(plot_folder="plot", output_folder="gifs"):
    """
    Create GIFs for each (sigma, RPA) combination from PNG files in sigma subfolders,
    sorting frames by deg in ascending order.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Find all sigma subfolders: plot/sigma_*/
    sigma_dirs = sorted(glob.glob(os.path.join(plot_folder, "sigma_*")))

    if not sigma_dirs:
        print(f"No sigma subfolders found in '{plot_folder}'.")
        return

    for sigma_dir in sigma_dirs:
        sigma_tag = os.path.basename(sigma_dir)  # e.g. "sigma_0.1"

        rpa_images = defaultdict(list)

        for png_file in glob.glob(os.path.join(sigma_dir, "*.png")):
            filename = os.path.basename(png_file)
            match = re.search(r"shape_rpa_(\d+\.?\d*)_phi_(\d+\.?\d*)", filename, re.IGNORECASE)
            if match:
                rpa = match.group(1)
                deg = float(match.group(2))
                rpa_images[rpa].append((deg, png_file))

        for rpa in sorted(rpa_images.keys()):
            sorted_images = sorted(rpa_images[rpa], key=lambda x: x[0])
            images = [Image.open(p) for _, p in sorted_images]

            if images:
                output_path = os.path.join(output_folder, f"{sigma_tag}_rpa_{rpa}.gif")
                images[0].save(
                    output_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=500,
                    loop=0,
                )
                print(f"Created: {output_path}")


if __name__ == "__main__":
    create_gifs_from_pngs()

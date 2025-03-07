import os


def load_paths_from_file(file_path):
    """
    Reads image paths from a text file, each path on a new line.

    :param file_path: Path to the text file containing image paths.
    :return: List of image paths.
    """
    with open(file_path, 'r') as file:
        paths = [line.strip() for line in file if line.strip()]
    return paths


def list_png_files(main_folder):
    """
    List all the .png files in the specified main folder, including those in subfolders.

    :param main_folder: The main folder to search in.
    :return: A list of paths to .nii.gz files.
    """
    png_files = []
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.png'):
                png_files.append(os.path.join(root, file))
    return png_files

def list_nii_files(main_folder):
    """
    List all the .nii.gz files in the specified main folder, including those in subfolders.

    :param main_folder: The main folder to search in.
    :return: A list of paths to .nii.gz files.
    """
    png_files = []
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.nii.gz'):
                png_files.append(os.path.join(root, file))
    return png_files

if __name__ == "__main__":
    data_dir = ''
    files = list_png_files(data_dir)
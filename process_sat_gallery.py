#将cvusa的satellite images 与 University-1652的satellite gallery 融合
import os
from shutil import copyfile

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

dir = '/home/wangtyu/datasets/CVUSA/val/satellite'
dst = '/home/wangtyu/datasets/University-Release/test/gallery_sat_usa_un/1652'
for d in os.listdir(dir):
    if os.path.isdir(os.path.join(dir, d)):
        f_path = os.path.join(dir, d)
        for root, _, fnames in sorted(os.walk(f_path)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    path = os.path.join(root, fname)
                    copyfile(path, os.path.join(dst,fname))


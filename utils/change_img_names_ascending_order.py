import os
import sys

formats = ["jpg", "jpeg", "png"]
img_num = 1
img_files = []

def get_all_files_ext(path, ext, sort=True):
  '''Get all files with the given file extension in the given path'''
  ext = "." + str(ext)
  files = [
      os.path.join(path, file_name)
      for file_name in os.listdir(path)
      if file_name.endswith(ext)
  ]

  if sort == True:
    return sorted(files)
  else:
    return files

# User Input Check
assert len(sys.argv) > 1, "Please write path contains images."
if (str(sys.platform) == "darwin") or (str(sys.platform) == "linux"):    # MacOS or Linux Environment
    arg_dir = str(sys.argv[1]) + "/"
else:   # Windows Environment
    arg_dir = str(sys.argv[1]) + "\\"

if len(sys.argv) > 2:   # Start with index
    img_num = int(sys.argv[2])
# User Input End

for format in formats:
   files = get_all_files_ext(arg_dir, format, sort=False)
   img_files = img_files + files

assert (len(img_files) > 0), "There is no image format file. Please insert files have 'jpeg', 'jpg', 'png' formats to the folder."

# Iterate
for img_file in img_files:
    print(img_file)
    ext = img_file.split(".")[-1]
    print(f"{img_file} is converted to the {img_num}.{ext}")
    oldName = os.path.join(arg_dir, img_file)
    # n = os.path.splitext(img_file)[0]  # The front part of the extension
    newName = str(img_num) + '.' + ext
    newName = os.path.join(arg_dir, newName)

    # Rename the file
    os.rename(oldName, newName)
    img_num += 1


# Example Usage
# /path/to/Python/Python37/python.exe /path/to/change_img_names_ascending_order.py path/to/target 
#
# Optional: Start image names with any number
#
# /path/to/Python/Python37/python.exe /path/to/change_img_names_ascending_order.py path/to/target 20
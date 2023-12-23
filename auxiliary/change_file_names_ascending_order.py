import os
import sys

formats = ["jpg", "jpeg", "png", "xml"]
img_num = 1
img_files = []
zero_pad = 1  # default zero padding for empty digits
newName = ""
oldName = ""

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

assert (len(img_files) > 0), "There is no image format file. Please insert files have 'jpeg', 'jpg', 'png', '.xml formats to the folder."

# Zero padding adjustment (How many zeros will be added to empty digits: 0001)
maximum_img_digit = len(str(len(img_files)))

if maximum_img_digit > zero_pad:
   zero_pad = maximum_img_digit + 2

# Iterate files and change names
for img_file in img_files:
    ext = img_file.split(".")[-1]
    img_name_len = len(str(img_num))
    oldName = os.path.join(arg_dir, img_file)
    newName = ""

    for _ in range(zero_pad - img_name_len):
       newName = newName + "0"

    newName = newName + str(img_num) + '.' + ext
    newName = os.path.join(arg_dir, newName)

    print(f"{img_file} is converted to the {img_num}.{ext}")

    # Rename the file
    os.rename(oldName, newName)
    img_num += 1


# Example Usage
# /path/to/Python/Python37/python.exe /path/to/change_img_names_ascending_order.py path/to/target 
#
# Optional: Start image names with any number
#
# /path/to/Python/Python37/python.exe /path/to/change_img_names_ascending_order.py path/to/target 20

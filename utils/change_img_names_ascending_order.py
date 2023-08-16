import os
import sys

formats = ["jpg", "jpeg", "png"]
arg_format = formats[0]
img_num = 1

# User Input Check
assert len(sys.argv) > 1, "Please write path contains images."
arg_dir = str(sys.argv[1]) + "\\"
if len(sys.argv) > 2:
    arg_format = str(sys.argv[2])
    assert (arg_format in formats), "Invalid target image format. Please choose one of 'jpeg', 'jpg', 'png' formats."
# User Input End

# Iterate
for file in os.listdir(arg_dir):
    print(f"{file} is converted to the {img_num}.{arg_format}")
    oldName = os.path.join(arg_dir, file)
    # n = os.path.splitext(file)[0]  # The front part of the extension
    newName = str(img_num) + '.' + arg_format
    newName = os.path.join(arg_dir, newName)

    # Rename the file
    os.rename(oldName, newName)
    img_num += 1

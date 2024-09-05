import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def sort_by_number(file_name):
    Initial = int(file_name.split("_")[1])
    return Initial

home_path = os.path.join(ROOT_DIR, "backbone_network\\data")
image_files = []
os.chdir(os.path.join(home_path, "Aug_train_data"))
for filename in sorted(os.listdir(os.getcwd()), key=sort_by_number):
    if filename.endswith(".txt"):
        image_files.append(filename)
    
os.chdir("..")
with open("train_ids.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()


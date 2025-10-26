from pathlib import Path
from import_slide import main

# Define the folder where the files are stored
folder = Path("d:/temp")
outfolder = Path("c:/pyproj/conf")

#2580 to 2677
# Example: files named "file1.txt", "file2.txt", ..., "file10.txt"
for i in range(2580, 2677):  # adjust range as needed
    infile = f"IMG_{i}.JPG"
    outfile = f"IMG_{i}.PNG"
    outfile_path = outfolder / outfile
    file_path = folder / infile

    if file_path.exists():
        print(infile)
        main(file_path, outfile_path)
    else:
        print(f"{file_path} does not exist")


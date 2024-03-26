import os
import sys

def compare_folders(folder1, folder2):
    # Get the list of files in both folders
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # Find missing files in folder1
    missing_in_folder1 = [f for f in files2 if f not in files1]

    # Find missing files in folder2
    missing_in_folder2 = [f for f in files1 if f not in files2]

    if missing_in_folder1:
        print(f"Files missing in {folder1}:")
        for file in missing_in_folder1:
            print(file)
        print()

    if missing_in_folder2:
        print(f"Files missing in {folder2}:")
        for file in missing_in_folder2:
            print(file)
        print()

    if not missing_in_folder1 and not missing_in_folder2:
        print("Both folders contain the same files.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <folder1> <folder2>")
        sys.exit(1)

    folder1 = sys.argv[1]
    folder2 = sys.argv[2]

    if not os.path.isdir(folder1):
        print(f"{folder1} is not a valid directory.")
        sys.exit(1)

    if not os.path.isdir(folder2):
        print(f"{folder2} is not a valid directory.")
        sys.exit(1)

    compare_folders(folder1, folder2)
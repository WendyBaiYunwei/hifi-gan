import os

# Specify the directory path where the filenames are located
directory_path = '/home/yunwei/new/VBeautifier/cpop-22k/cpop_singer'

# Specify the name of the output .txt file
output_file = 'LJSpeech-1.1/training.txt'

# Create or open the output .txt file in write mode
with open(output_file, 'w') as file:
    # Walk through the directory tree
    for root, _, files in os.walk(directory_path):
        for filename in files:
            # Get the absolute path of the file
            file_path = os.path.join(root, filename)
            # Write the relative path of the file followed by '|' to the file
            # relative_path = os.path.relpath(file_path, directory_path)
            file_path = file_path.replace('.wav', '')
            file.write(file_path + '|\n')

print(f"Filenames from files in the directory and nested directories have been written to {output_file}.")

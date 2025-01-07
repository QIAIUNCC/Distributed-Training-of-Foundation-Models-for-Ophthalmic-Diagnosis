import pandas as pd
import os
import glob

def correct_directory_names(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Check if 'directory' column exists
    if 'Directory' in df.columns:
        # Replace 'NOrmal' with 'Normal' in the 'directory' column
        df['Directory'] = df['Directory'].apply(lambda x: x.replace('NOrmal', 'Normal') if isinstance(x, str) else x)

    # Save the corrected DataFrame back to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Corrected CSV saved to {output_csv}")


def rename_images_in_directory(root_path):
    # Recursively find all image files in the directory and subdirectories
    image_files = glob.glob(os.path.join(root_path, '**', '**','*.*'), recursive=True)

    # Iterate through each file
    for file_path in image_files:
        # Check if the filename contains 'NOrmal'
        directory, filename = os.path.split(file_path)
        if 'NOrmal' in filename:
            # Replace 'NOrmal' with 'Normal' in the filename
            new_filename = filename.replace('NOrmal', 'Normal')
            new_file_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"Renamed: {file_path} -> {new_file_path}")

# Example usage
input_csv = '/data1/OCT/NEH_UT_2021RetinalOCTDataset/data_information.csv'
# output_csv = 'output.csv'
# correct_directory_names(input_csv, output_csv)

rename_images_in_directory("/data1/OCT/NEH_UT_2021RetinalOCTDataset")

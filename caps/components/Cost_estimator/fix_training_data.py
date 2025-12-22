import os

def replace_in_files(directory, old_string, new_string):
    """
    Replaces a specific string in all files within a directory.

    :param directory: Path to the directory containing files to process.
    :param old_string: String to replace.
    :param new_string: Replacement string.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Replace old string with new string
                updated_content = content.replace(old_string, new_string)

                # Write the updated content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)

                print(f"Processed file: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    # Set the directory path and strings to replace
    directory_path = "C:/Users/adoko/PycharmProjects/autoPipe/Cost_estimator/experiments/paper/"
    replace_in_files(directory_path, "fit,True", "fit")

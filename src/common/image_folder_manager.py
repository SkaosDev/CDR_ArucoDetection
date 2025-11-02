import os

def clear_image_folder(folder_name):
    """
    Remove all files in the specified folder.
    """
    if not os.path.exists(folder_name):
        print(f"Folder '{folder_name}' don't exist.")
        return

    for filename in os.listdir(folder_name):
        file_path = os.path.join(folder_name, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error: unable to remove image at {file_path}: {e}")

def clear_old_images(folder_name, max_images_in_folder):
    """
    Remove oldest images in the specified folder
    until the number of images is less than or equal to max_images_in_folder.
    """
    if not os.path.exists(folder_name):
        print(f"Error: Folder '{folder_name}' don't exist.")
        return

    files = [os.path.join(folder_name, f) for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f))]
    files.sort(key=os.path.getmtime)

    while len(files) > max_images_in_folder:
        file_to_remove = files.pop(0)
        try:
            os.remove(file_to_remove)
        except Exception as e:
            print(f"Error: unable to remove image at {file_to_remove}: {e}")
import os

def clear_image_folder(folder_name):
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

    print(f"All file in '{folder_name}' have been deleted.")

def clear_old_images(folder_name, max_images_in_folder):
    if not os.path.exists(folder_name):
        print(f"Error: Folder '{folder_name}' don't exist.")
        return

    files = [os.path.join(folder_name, f) for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f))]
    files.sort(key=os.path.getmtime)

    while len(files) > max_images_in_folder:
        file_to_remove = files.pop(0)
        try:
            os.remove(file_to_remove)
            print(f"Removed old image: {file_to_remove}")
        except Exception as e:
            print(f"Error: unable to remove image at {file_to_remove}: {e}")
import os
import sys
from PIL import Image


def concat_images_horizontally(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return

    # Get all image files in the folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and
                   os.path.splitext(f)[1].lower() in image_extensions]

    # Check if there are any images
    if not image_files:
        print(f"Error: No images found in the folder '{folder_path}'.")
        return

    # Sort files in reverse alphabetical order
    image_files.sort(reverse=True)

    # Open all images
    images = []
    for img_file in image_files:
        try:
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path)
            images.append(img)
            print(f"Loaded: {img_file}")
        except Exception as e:
            print(f"Error loading {img_file}: {e}")

    # Check if we successfully loaded any images
    if not images:
        print("Error: Could not load any images.")
        return

    # Calculate the width and height of the final image
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new blank image with the calculated dimensions
    result = Image.new('RGB', (total_width, max_height))

    # Paste the images side by side
    current_width = 0
    for img in images:
        # Paste at the current position
        result.paste(img, (current_width, 0))
        current_width += img.width

    # Generate output filename
    output_path = os.path.join(folder_path, "concatenated_images.jpg")

    # Save the result
    result.save(output_path)
    print(f"Concatenated image saved as: {output_path}")
    return output_path


if __name__ == "__main__":

    folder_path = os.path.join(os.getcwd(), "YYY")
    concat_images_horizontally(folder_path)

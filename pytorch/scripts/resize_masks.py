import os
from PIL import Image

def resize_images(input_dir, output_dir, width=640, height=480):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            try:
                # Open the image
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path)
                
                # Resize image while maintaining aspect ratio
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                
                # Save the resized image to the output directory
                output_path = os.path.join(output_dir, filename)
                img.save(output_path)
                print(f"Resized and saved: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        else:
            print(f"Skipped: {filename} (unsupported format)")

if __name__ == "__main__":
    input_directory = "../seame_masks" 
    output_directory = "../seame_new_masks" 
    
    resize_images(input_directory, output_directory)
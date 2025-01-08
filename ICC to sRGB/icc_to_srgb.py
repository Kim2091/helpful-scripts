import os
import argparse
from PIL import Image
import PIL.ImageCms as ImageCms
from io import BytesIO

def process_image(input_path, output_path):
    """Process a single image by applying its ICC profile and converting to sRGB while preserving alpha."""
    try:
        # Open the image
        img = Image.open(input_path)
        
        # Store alpha channel if it exists
        has_alpha = 'A' in img.getbands()
        if has_alpha:
            # Extract alpha channel
            alpha = img.split()[-1]
            # Convert to RGB for color processing
            rgb_img = img.convert('RGB')
        else:
            rgb_img = img
            
        # Check if image has an ICC profile
        if 'icc_profile' in img.info:
            # Create profile objects
            input_profile = ImageCms.ImageCmsProfile(BytesIO(img.info['icc_profile']))
            srgb_profile = ImageCms.createProfile('sRGB')
            
            # Convert the image
            rgb_converted = ImageCms.profileToProfile(
                rgb_img,
                input_profile,
                srgb_profile,
                outputMode='RGB'
            )
        else:
            # If no ICC profile, just use RGB conversion
            rgb_converted = rgb_img
            
        # Reapply alpha channel if it existed
        if has_alpha:
            channels = list(rgb_converted.split())
            channels.append(alpha)
            final_image = Image.merge('RGBA', channels)
        else:
            final_image = rgb_converted
            
        # Save the converted image
        final_image.save(output_path, 'PNG', icc_profile=None)
        print(f"Processed: {input_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def process_folder(input_folder, output_folder):
    """Process all images in a directory."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each file in input directory
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, os.path.splitext(relative_path)[0] + '.png')
                
                # Create necessary subdirectories in output folder
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                process_image(input_path, output_path)

def process_input(input_path, output_path):
    """Process a single image or a folder of images."""
    if os.path.isfile(input_path):
        # Process a single image
        process_image(input_path, output_path)
    elif os.path.isdir(input_path):
        # Process a folder of images
        process_folder(input_path, output_path)
    else:
        print(f"Invalid input: {input_path} is neither a file nor a directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert images with ICC profiles to sRGB.')
    parser.add_argument('input_path', type=str, help='Input file or folder containing images')
    parser.add_argument('output_path', type=str, help='Output file or folder to save converted images')
    args = parser.parse_args()

    process_input(args.input_path, args.output_path)

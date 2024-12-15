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
        if has_alpha:
            final_image.save(output_path, 'PNG', icc_profile=None)
        else:
            final_image.save(output_path, 'PNG', icc_profile=None)
            
        print(f"Processed: {input_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def process_directory(input_dir, output_dir):
    """Process all images in a directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    supported_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    
    # Process each file in input directory
    for filename in os.listdir(input_dir):
        # Check if file is an image
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_extensions:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
            process_image(input_path, output_path)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Apply embedded ICC profile and convert to sRGB')
    parser.add_argument('-i', '--input', required=True, help='Input image or directory')
    parser.add_argument('-o', '--output', required=True, help='Output image or directory')
    
    args = parser.parse_args()
    
    # Check if input is a directory or single file
    if os.path.isdir(args.input):
        process_directory(args.input, args.output)
    else:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        process_image(args.input, args.output)

if __name__ == "__main__":
    main()
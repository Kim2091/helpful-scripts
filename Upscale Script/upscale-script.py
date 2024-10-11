import os
import configparser
import torch
from PIL import Image
import spandrel
import spandrel_extra_arches
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import traceback
import gc
import argparse
import sys
from tqdm import tqdm
import chainner_ext

# Install extra architectures
spandrel_extra_arches.install()

# Read configuration
config = configparser.ConfigParser()
config.read('config.ini')

TILE_SIZE = config['Processing'].get('TileSize', '512').lower()
PRECISION = config['Processing'].get('Precision', 'auto').lower()
THREAD_POOL_WORKERS = int(config['Processing'].get('ThreadPoolWorkers', 1))
OUTPUT_FORMAT = config['Processing'].get('OutputFormat', 'png').lower()
ALPHA_HANDLING = config['Processing'].get('AlphaHandling', 'resize').lower()
GAMMA_CORRECTION = config['Processing'].getboolean('GammaCorrection', False)

# Create a ThreadPoolExecutor for running CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)

# Supported image formats
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.webp', '.tga', '.bmp', '.tiff')

def upscale_tensor(img_tensor, model, tile_size):
    _, _, h, w = img_tensor.shape
    output_h, output_w = h * model.scale, w * model.scale

    output_dtype = torch.float32 if PRECISION == 'fp32' else torch.float16
    output_tensor = torch.zeros((1, img_tensor.shape[1], output_h, output_w), dtype=output_dtype, device='cuda')

    if tile_size == "native":
        tile_size = max(h, w)

    tile_size = int(tile_size)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img_tensor[:, :, y:min(y+tile_size, h), x:min(x+tile_size, w)]

            with torch.inference_mode():
                if model.supports_bfloat16 and PRECISION in ['auto', 'bf16']:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        upscaled_tile = model(tile)
                elif model.supports_half and PRECISION in ['auto', 'fp16']:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        upscaled_tile = model(tile)
                else:
                    upscaled_tile = model(tile)

            output_tensor[:, :, y*model.scale:min((y+tile_size)*model.scale, output_h),
                          x*model.scale:min((x+tile_size)*model.scale, output_w)].copy_(upscaled_tile)

    return output_tensor

def load_model(model_path):
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    
    try:
        model = spandrel.ModelLoader().load_from_file(model_path)
        if isinstance(model, spandrel.ImageModelDescriptor):
            return model.cuda().eval()
        else:
            raise ValueError(f"Invalid model type for {model_path}")
    except Exception as e:
        print(f"Failed to load model {model_path}: {str(e)}")
        raise

def upscale_image(image, model, tile_size, alpha_handling, gamma_correction):
    has_alpha = image.mode == 'RGBA'
    if has_alpha:
        rgb_image, alpha = image.convert('RGB'), image.split()[3]
    else:
        rgb_image = image

    # Upscale RGB
    rgb_tensor = torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).cuda()
    upscaled_rgb_tensor = upscale_tensor(rgb_tensor, model, tile_size)
    upscaled_rgb = Image.fromarray((upscaled_rgb_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    if has_alpha:
        if alpha_handling == 'upscale':
            # Create a 3-channel tensor from the alpha channel
            alpha_array = np.array(alpha)
            alpha_3channel = np.stack([alpha_array, alpha_array, alpha_array], axis=2)
            alpha_tensor = torch.from_numpy(alpha_3channel).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).cuda()
            
            # Upscale the 3-channel alpha tensor
            upscaled_alpha_tensor = upscale_tensor(alpha_tensor, model, tile_size)
            
            # Extract a single channel from the result
            upscaled_alpha = Image.fromarray((upscaled_alpha_tensor[0, 0].cpu().numpy() * 255).astype(np.uint8))
        elif alpha_handling == 'resize':
            # Resize alpha using chainner_ext.resize with CubicMitchell filter
            alpha_np = np.array(alpha, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            alpha_np = alpha_np.reshape(alpha_np.shape[0], alpha_np.shape[1], 1)  # Add channel dimension
            upscaled_alpha_np = chainner_ext.resize(
                alpha_np,
                (upscaled_rgb.width, upscaled_rgb.height),
                chainner_ext.ResizeFilter.CubicMitchell,
                gamma_correction=gamma_correction
            )
            # Convert back to 0-255 range and clip values
            upscaled_alpha_np = np.clip(upscaled_alpha_np * 255, 0, 255)
            upscaled_alpha = Image.fromarray(upscaled_alpha_np.squeeze().astype(np.uint8))
        elif alpha_handling == 'discard':
            # Discard alpha
            return upscaled_rgb
        
        # Merge upscaled RGB and alpha
        upscaled_rgba = upscaled_rgb.copy()
        upscaled_rgba.putalpha(upscaled_alpha)
        return upscaled_rgba
    else:
        return upscaled_rgb
                
def process_image(input_path, output_path, model):
    try:
        image = Image.open(input_path)

        start_time = time.time()

        result = upscale_image(image, model, TILE_SIZE, ALPHA_HANDLING, GAMMA_CORRECTION)

        upscale_time = time.time() - start_time

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        result.save(output_path, OUTPUT_FORMAT.upper())

        save_time = time.time() - start_time - upscale_time
        total_time = time.time() - start_time

        return total_time

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        traceback.print_exc()
        return None

def process_directory(input_dir, output_dir, model):
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                image_files.append((root, file))

    print(f"Found {len(image_files)} image(s) to process.")

    total_time = 0
    successful_images = 0

    with tqdm(total=len(image_files), desc="Processing Images", unit="image") as pbar:
        for root, file in image_files:
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            output_path = os.path.splitext(output_path)[0] + f'.{OUTPUT_FORMAT}'
            
            processing_time = process_image(input_path, output_path, model)
            
            if processing_time is not None:
                total_time += processing_time
                successful_images += 1
            
            pbar.update(1)

    print(f"Successfully processed {successful_images} out of {len(image_files)} images.")
    print(f"Total processing time: {total_time:.2f} seconds")
    if successful_images > 0:
        print(f"Average time per image: {total_time / successful_images:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Image Upscaling Tool")
    parser.add_argument("--input", required=True, help="Input image file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", required=True, help="Path to the model file")
    args = parser.parse_args()

    print(f"Input path: {args.input}")
    print(f"Output path: {args.output}")
    print(f"Model path: {args.model}")
    print(f"Tile size: {TILE_SIZE}")
    print(f"Output format: {OUTPUT_FORMAT}")

    if not os.path.exists(args.input):
        print(f"Error: Input path not found: {args.input}")
        return

    if not os.path.exists(args.output):
        print(f"Creating output directory: {args.output}")
        os.makedirs(args.output)

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return

    try:
        print("Loading model...")
        model = load_model(args.model)
        print("Model loaded successfully.")

        if os.path.isfile(args.input):
            print(f"Processing single file: {args.input}")
            output_path = os.path.join(args.output, os.path.basename(args.input))
            output_path = os.path.splitext(output_path)[0] + f'.{OUTPUT_FORMAT}'
            process_image(args.input, output_path, model)
        else:
            print(f"Processing directory: {args.input}")
            process_directory(args.input, args.output, model)

        print("All processing completed.")

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        print("Cleanup completed.")

if __name__ == "__main__":
    main()
    print("Script execution finished.")

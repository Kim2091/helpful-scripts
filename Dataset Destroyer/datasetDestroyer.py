import configparser
import os
import cv2
import time
import numpy as np
import ffmpeg
from random import random, randint, choice, shuffle, uniform
import concurrent.futures
from tqdm import tqdm
from PIL import Image, ImageFilter
from chainner_ext import DiffusionAlgorithm, UniformQuantization, error_diffusion_dither, resize, ResizeFilter

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

# Get config values
input_folder = config.get('main', 'input_folder')
output_folder = config.get('main', 'output_folder')
output_format = config.get('main', 'output_format')
degradations = config.get('main', 'degradations').split(',')
degradations_randomize = config.getboolean('main', 'randomize')
blur_algorithms = config.get('blur', 'algorithms').split(',')
blur_randomize = config.getboolean('blur', 'randomize')
blur_range = tuple(map(int, config.get('blur', 'range').split(',')))
noise_algorithms = config.get('noise', 'algorithms').split(',')
noise_randomize = config.getboolean('noise', 'randomize')
noise_range = tuple(map(int, config.get('noise', 'range').split(',')))
noise_scale_factor = config.getfloat('noise', 'scale_factor')
sp_noise_range = tuple(map(int, config.get('noise', 'sp_range').split(',')))  # Salt and pepper noise range
sp_noise_scale_factor = config.getfloat('noise', 'sp_scale_factor')  # Salt and pepper noise scale factor
compression_algorithms = config.get('compression', 'algorithms').split(',')
compression_randomize = config.getboolean('compression', 'randomize')
jpeg_quality_range = tuple(map(int, config.get('compression', 'jpeg_quality_range').split(',')))
webp_quality_range = tuple(map(int, config.get('compression', 'webp_quality_range').split(',')))
h264_crf_level_range = tuple(map(int, config.get('compression', 'h264_crf_level_range').split(',')))
hevc_crf_level_range = tuple(map(int, config.get('compression', 'hevc_crf_level_range').split(',')))
vp9_crf_level_range = tuple(int(x) for x in config.get('compression', 'vp9_crf_level_range').split(','))
mpeg_bitrate_range = tuple(map(int, config.get('compression', 'mpeg_bitrate_range').split(',')))
mpeg2_bitrate_range = tuple(map(int, config.get('compression', 'mpeg2_bitrate_range').split(',')))
size_factor = config.getfloat('scale', 'size_factor')
scale_algorithms = config.get('scale', 'algorithms').split(',')
down_up_scale_algorithms = config.get('scale', 'down_up_algorithms').split(',')
scale_randomize = config.getboolean('scale', 'randomize')
scale_range = tuple(map(float, config.get('scale', 'range').split(',')))
blur_scale_factor = config.getfloat('blur', 'scale_factor')
unsharp_mask_radius_range = tuple(map(float, config.get('unsharp_mask', 'radius_range').split(',')))
unsharp_mask_percent_range = tuple(map(float, config.get('unsharp_mask', 'percent_range').split(',')))
unsharp_mask_threshold_range = tuple(map(int, config.get('unsharp_mask', 'threshold_range').split(',')))
print_to_image = config.getboolean('main', 'print')
print_to_textfile = config.getboolean('main', 'textfile')
path_to_textfile = config.get('main', 'textfile_path')

# Add config values for quantization
quantization_algorithms = config.get('quantization', 'algorithms').split(',')
quantization_randomize = config.getboolean('quantization', 'randomize')
quantization_range = tuple(map(int, config.get('quantization', 'range').split(',')))

def print_text_to_image(image, text, order):
    h, w = image.shape[:2]
    font_scale = w / 1200
    font_thickness = int(font_scale * 2)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_width, text_height = text_size
    x = 10
    y = int(order * text_height * 1.5) + 10
    return cv2.putText(image, f"{order}. {text}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (255, 0, 0), font_thickness, cv2.LINE_AA)

# Append given text as a new line at the end of file (if file not exists it creates and inserts line, otherwise it just appends newline)               
def print_text_to_textfile(file_name, text_to_append):
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)
        
def apply_blur(image):
    text = ''
    # Choose blur algorithm
    if blur_randomize:
        algorithm = choice(blur_algorithms)
    else:
        algorithm = blur_algorithms[0]

    # Normalize the image to the range [0, 1]
    image = image.astype(float) / 255

    # Apply blur with chosen algorithm
    if algorithm == 'average':
        ksize = randint(*blur_range)
        ksize = int(ksize * blur_scale_factor)  # Scale down ksize by blur_scale_factor
        ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure ksize is an odd integer
        image = cv2.blur(image, (ksize, ksize))
        text = f"{algorithm} ksize={ksize}"
    elif algorithm == 'gaussian':
        ksize = randint(*blur_range) | 1
        ksize = int(ksize * blur_scale_factor)  # Scale down ksize by blur_scale_factor
        ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure ksize is an odd integer
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        text = f"{algorithm} ksize={ksize}"
    elif algorithm == 'anisotropic':
        # Apply anisotropic blur using a Gaussian filter with different standard deviations in the x and y directions
        sigma_x = randint(*blur_range)
        sigma_y = randint(*blur_range)
        angle = uniform(0, 360)

        # Scale down sigma by blur_scale_factor
        sigma_x *= blur_scale_factor  
        sigma_y *= blur_scale_factor

        # Convert angle to radians
        angle = np.deg2rad(angle)

        # Create a 2D Gaussian kernel with the desired direction
        kernel_size = max(2 * int(4 * max(sigma_x, sigma_y) + 0.5) + 1, 3)
        y, x = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, -kernel_size//2 + 1:kernel_size//2 + 1]
        rotx = x * np.cos(angle) - y * np.sin(angle)  # Rotate x by the angle
        roty = x * np.sin(angle) + y * np.cos(angle)  # Rotate y by the angle
        kernel = np.exp(-(rotx**2/(2*sigma_x**2) + roty**2/(2*sigma_y**2)))

        # Normalize the kernel
        kernel /= np.sum(kernel)

        # Apply the kernel to the image
        image = cv2.filter2D(image, -1, kernel)

        text = f"{algorithm} sigma_x={sigma_x} sigma_y={sigma_y} angle={np.rad2deg(angle)}"

    # Scale the image back to the range [0, 255]
    image = (image * 255).astype(np.uint8)

    return image, text

def apply_noise(image):
    # Normalize the image to the range [0, 1]
    image = image.astype(float) / 255

    text = ''
    # Choose noise algorithm
    if noise_randomize:
        algorithm = choice(noise_algorithms)
    else:
        algorithm = noise_algorithms[0]

    # Apply noise with chosen algorithm
    if algorithm == 'uniform':
        intensity = randint(*noise_range)
        intensity *= noise_scale_factor  # Scale down intensity by noise_scale_factor
        noise = np.random.uniform(-intensity, intensity, image.shape)
        image += noise
        text = f"{algorithm} intensity={intensity}"

    elif algorithm == 'gaussian':
        mean = 0
        var = randint(*noise_range)
        var *= noise_scale_factor # Scale down variance by noise_scale_factor
        sigma = var**0.5
        noise = np.random.normal(mean, sigma, image.shape)
        image += noise
        text = f"{algorithm} variance={var}"

    elif algorithm == 'color':
        noise = np.zeros_like(image)
        m = (0, 0, 0)
        s = (randint(*noise_range) * noise_scale_factor, randint(*noise_range) * noise_scale_factor, randint(*noise_range) * noise_scale_factor)
        cv2.randn(noise, m, s)
        image += noise
        text = f"{algorithm} s={s}"

    elif algorithm == 'gray':
        gray_noise = np.zeros((image.shape[0], image.shape[1]))
        m = (0,)
        s = (randint(*noise_range) * noise_scale_factor,)
        cv2.randn(gray_noise, m, s)
        for i in range(image.shape[2]):  # Add noise to each channel separately
            image[..., i] += gray_noise
        text = f"{algorithm} s={s}"

    elif algorithm == 'simu_iso':
        # Simulated ISO noise based on a combination of Gaussian and Poisson noise
        mean = 0
        var = randint(*noise_range)
        var *= noise_scale_factor # Scale down variance by noise_scale_factor
        sigma = var**0.5
        gaussian = np.random.normal(mean, sigma, image.shape)
        poisson = np.random.poisson(image)
        noise = gaussian + poisson
        # Scale the noise to the range of the image
        noise = noise - np.min(noise)
        noise = noise / np.max(noise)
        image += noise
        text = f"{algorithm} variance={var}"

    elif algorithm == 'salt-and-pepper':
        # Salt-and-pepper noise
        intensity = randint(*sp_noise_range)
        intensity *= sp_noise_scale_factor  # Scale down intensity by sp_noise_scale_factor

        # Pepper mode
        num_pepper = np.ceil(intensity * image.size * 0.25)  # Reduced to 25% of the image size
        x_pepper = np.random.randint(0, image.shape[1], int(num_pepper))
        y_pepper = np.random.randint(0, image.shape[0], int(num_pepper))
        image[y_pepper, x_pepper] = 0

        # Salt mode
        num_salt = np.ceil(intensity * image.size * 0.5)
        x_salt = np.random.randint(0, image.shape[1], int(num_salt))
        y_salt = np.random.randint(0, image.shape[0], int(num_salt))
        image[y_salt, x_salt] = 1
        text = f"{algorithm} intensity={intensity}"

    # Clip the values to the range [0, 1] and scale back to the range [0, 255]
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    return image, text

def apply_quantization(image):
    # Assert that the input image has 3 dimensions
    assert len(image.shape) == 3, "Input image must have 3 dimensions (height, width, channels)"

    text = ''
    # Choose quantization algorithm
    if quantization_randomize:
        algorithm = choice(quantization_algorithms)
    else:
        algorithm = quantization_algorithms[0]

    # Map string algorithm names to DiffusionAlgorithm enum values
    algorithm_mapping = {
        'floyd_steinberg': DiffusionAlgorithm.FloydSteinberg,
        'jarvis_judice_ninke': DiffusionAlgorithm.JarvisJudiceNinke,
        'stucki': DiffusionAlgorithm.Stucki,
        'atkinson': DiffusionAlgorithm.Atkinson,
        'burkes': DiffusionAlgorithm.Burkes,
        'sierra': DiffusionAlgorithm.Sierra,
        'two_row_sierra': DiffusionAlgorithm.TwoRowSierra,
        'sierra_lite': DiffusionAlgorithm.SierraLite,
    }

    # Apply quantization with chosen algorithm
    if algorithm in algorithm_mapping:
        colors_per_channel = randint(*quantization_range)
        quant = UniformQuantization(colors_per_channel=colors_per_channel)
        image_np = np.array(image).astype(np.float32) / 255.0

        # Apply the chosen dithering algorithm to each color channel separately
        for i in range(image_np.shape[2]):
            dithered_channel = error_diffusion_dither(image_np[..., i], quant, algorithm_mapping[algorithm])
            # Reshape the output to (height, width) if necessary
            if len(dithered_channel.shape) == 3:
                dithered_channel = dithered_channel.squeeze(-1)
            image_np[..., i] = dithered_channel

        # Convert the numpy array back to an image
        dithered_image_np = np.round(image_np * 255).astype(np.uint8)  # Round before converting to uint8
        image = Image.fromarray(dithered_image_np)

        text = f"{algorithm} colors_per_channel={colors_per_channel}"
    else:
        raise ValueError(f"Unsupported quantization algorithm: {algorithm}")

    # Convert the image back to a numpy array before returning
    image = np.array(image)

    return image, text
    
def apply_unsharp_mask(image, config):
    text = ''
    # Choose unsharp mask parameters
    radius = np.random.uniform(unsharp_mask_radius_range[0], unsharp_mask_radius_range[1])
    percent = np.random.uniform(unsharp_mask_percent_range[0], unsharp_mask_percent_range[1])
    threshold = np.random.randint(unsharp_mask_threshold_range[0], unsharp_mask_threshold_range[1])

    # Apply unsharp mask with chosen parameters
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharpened = cv2.addWeighted(image, 1.0 + percent, blurred, -percent, threshold)
    image = np.clip(sharpened, 0, 255).astype(np.uint8)  # Clip values to 8-bit range

    text = f"unsharp_mask radius={radius} percent={percent} threshold={threshold}"

    return image, text

def apply_compression(image):
    text = ''
    # Choose compression algorithm
    if compression_randomize:
        algorithm = choice(compression_algorithms)
    else:
        algorithm = compression_algorithms[0]

    # Apply compression with chosen algorithm
    if algorithm == 'jpeg':
        quality = randint(*jpeg_quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        image = cv2.imdecode(encimg, 1).copy()
        text = f"{algorithm} quality={quality}"

    elif algorithm == 'webp':
        quality = randint(*webp_quality_range)
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        result, encimg = cv2.imencode('.webp', image, encode_param)
        image = cv2.imdecode(encimg, 1).copy()
        text = f"{algorithm} quality={quality}"

    elif algorithm in ['h264', 'hevc', 'mpeg', 'mpeg2', 'vp9']:
        # Convert image to video format
        height, width, _ = image.shape
        codec = algorithm
        container = 'mpeg'
        if algorithm == 'mpeg':
            codec = 'mpeg1video'

        elif algorithm == 'mpeg2':
            codec = 'mpeg2video'

        elif algorithm == 'vp9':
            codec = 'libvpx-vp9'
            container = 'webm'
            crf_level = randint(*vp9_crf_level_range)
            output_args = {'crf': str(crf_level), 'b:v': '0', 'cpu-used': '5'}
    
        # Get CRF level or bitrate from config
        if algorithm == 'h264':
            crf_level = randint(*h264_crf_level_range)
            output_args = {'crf': crf_level}

        elif algorithm == 'hevc':
            crf_level = randint(*hevc_crf_level_range)
            output_args = {'crf': crf_level, 'x265-params': 'log-level=0'}

        elif algorithm == 'mpeg':
            bitrate = str(randint(*mpeg_bitrate_range)) + 'k'
            output_args = {'b': bitrate}

        elif algorithm == 'mpeg2':
            bitrate = str(randint(*mpeg2_bitrate_range)) + 'k'
            output_args = {'b': bitrate}

        elif algorithm == 'vp9':
            codec = 'libvpx-vp9'
            container = 'webm'
            crf_level = randint(*vp9_crf_level_range)
            output_args = {'crf': str(crf_level), 'b:v': '0', 'cpu-used': '5'}

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Encode image using ffmpeg
        process1 = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}')
            .output('pipe:', format=container, vcodec=codec, **output_args)
            .global_args('-loglevel', 'fatal') # Disable error reporting because of buffer errors
            .global_args('-max_muxing_queue_size', '300000')
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )
        process1.stdin.write(image.tobytes())
        process1.stdin.flush()  # Ensure all data is written
        process1.stdin.close()

        # Add a delay between each image to help resolve buffer errors
        time.sleep(0.1) 
        
        # Decode compressed video back into image format using ffmpeg
        process2 = (
            ffmpeg
            .input('pipe:', format=container)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .global_args('-loglevel', 'fatal') # Disable error reporting because of buffer errors
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )

        out, err = process2.communicate(input=process1.stdout.read())

        process1.wait()

        try:
            image = np.frombuffer(out, np.uint8).reshape([height, width, 3]).copy()
            first_arg = list(output_args.items())[0]
            text = f"{algorithm} {first_arg[0]}={first_arg[1]}"
        except ValueError as e:
            logging.error(f'Error reshaping output from ffmpeg: {e}')
            logging.error(f'Image dimensions: {width}x{height}')
            logging.error(f'ffmpeg stderr output: {err}')
            raise e

    return image, text

def apply_scale(image):
    # Convert image to float32 and normalize pixel values
    image = np.float32(image) / 255.0

    text = ''
    # Calculate new size
    h, w = image.shape[:2]
    new_h = int(h * size_factor)
    new_w = int(w * size_factor)

    # Choose scale algorithm
    if scale_randomize:
        algorithm = choice(scale_algorithms)
    else:
        algorithm = scale_algorithms[0]

    interpolation_map = {
        'nearest': ResizeFilter.Nearest,
        'linear': ResizeFilter.Linear,
        'cubic_catrom': ResizeFilter.CubicCatrom,
        'cubic_mitchell': ResizeFilter.CubicMitchell,
        'cubic_bspline': ResizeFilter.CubicBSpline,
        'lanczos': ResizeFilter.Lanczos,
        'gauss': ResizeFilter.Gauss
    }

    # Disable gamma correction for nearest neighbor
    gamma_correction = algorithm != 'nearest'

    if algorithm == 'down_up':
        if scale_randomize:
            algorithm1 = choice(down_up_scale_algorithms)
            algorithm2 = choice(down_up_scale_algorithms)
        else:
            algorithm1 = down_up_scale_algorithms[0]
            algorithm2 = down_up_scale_algorithms[-1]
        scale_factor = np.random.uniform(*scale_range)
        image = resize(image, (int(w * scale_factor), int(h * scale_factor)), interpolation_map[algorithm1], gamma_correction=True)
        image = resize(image, (new_w, new_h), interpolation_map[algorithm2], gamma_correction=True)
        if print_to_image:
            text = f"{algorithm} scale1factor={scale_factor:.2f} scale1algorithm={algorithm1} scale2factor={size_factor/scale_factor:.2f} scale2algorithm={algorithm2}"
        if print_to_textfile:
            text = f"{algorithm} scale1factor={scale_factor:.2f} scale1algorithm={algorithm1} scale2factor={size_factor/scale_factor:.2f} scale2algorithm={algorithm2}"
    else:
        image = resize(image, (new_w, new_h), interpolation_map[algorithm], gamma_correction=True)
        if print_to_image:
            text = f"{algorithm} size factor={size_factor}"
        if print_to_textfile:
            text = f"{algorithm} size factor={size_factor}"

    # Convert image back to uint8 after resizing for script compatibility
    image = (image * 255).astype(np.uint8)

    return image, text
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return
    
    degradation_order = degradations.copy()
    all_text = []
    if degradations_randomize:
        shuffle(degradation_order)
    for order, degradation in enumerate(degradation_order, 1):
        if degradation == 'blur':
            image, text = apply_blur(image)
        elif degradation == 'noise':
            image, text = apply_noise(image)
        elif degradation == 'compression':
            image, text = apply_compression(image)
        elif degradation == 'scale':
            image, text = apply_scale(image)
        elif degradation == 'quantization':
            image, text = apply_quantization(image)
        elif degradation == 'unsharp_mask':
            image, text = apply_unsharp_mask(image, config)
        all_text.append(f"{degradation}: {text}")
    if print_to_image:
        for order, text in enumerate(all_text, 1):
            image = print_text_to_image(image, text, order)

    # Save image
    output_path = os.path.join(output_folder, os.path.relpath(image_path, input_folder))
    output_path = os.path.splitext(output_path)[0] + '.' + output_format
    
    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    

    if print_to_textfile:
        print_text_to_textfile(path_to_textfile + "/applied_degradations.txt",os.path.basename(output_path) + ' - ' + ', '.join(all_text))
# Process images recursively
image_paths = []
for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        image_paths.append(os.path.join(subdir, file))

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_image, image_path) for image_path in image_paths}
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        try:
            for f in tqdm(concurrent.futures.as_completed(futures), **kwargs):
                # Disable this block and replace with "pass" to hide exceptions
                try:
                    f.result()  # This will raise the exception if one was thrown
                except Exception as e:
                    print(f"An error occurred: {e}")

        except KeyboardInterrupt:
            print("Interrupted by user, terminating processes...")
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()


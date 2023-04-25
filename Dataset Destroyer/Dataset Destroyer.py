import configparser
import os
import cv2
import numpy as np
import ffmpeg
from random import randint, choice, shuffle
from tqdm import tqdm
from scipy import signal

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
compression_algorithms = config.get('compression', 'algorithms').split(',')
compression_randomize = config.getboolean('compression', 'randomize')
jpeg_quality_range = tuple(map(int, config.get('compression', 'jpeg_quality_range').split(',')))
webp_quality_range = tuple(map(int, config.get('compression', 'webp_quality_range').split(',')))
scale_algorithms = config.get('scale', 'algorithms').split(',')
scale_randomize = config.getboolean('scale', 'randomize')
scale_range = tuple(map(float, config.get('scale', 'range').split(',')))

def apply_blur(image):
    # Choose blur algorithm
    if blur_randomize:
        algorithm = choice(blur_algorithms)
    else:
        algorithm = blur_algorithms[0]

    # Apply blur with chosen algorithm
    if algorithm == 'average':
        ksize = randint(*blur_range)
        image = cv2.blur(image, (ksize, ksize))
    elif algorithm == 'gaussian':
        ksize = randint(*blur_range) | 1
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif algorithm == 'isotropic':
        # Apply isotropic blur using a Gaussian filter with the same standard deviation in both the x and y directions
        sigma = randint(*blur_range)
        ksize = 2 * int(4 * sigma + 0.5) + 1
        image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    elif algorithm == 'anisotropic':
        # Apply anisotropic blur using a Gaussian filter with different standard deviations in the x and y directions
        sigma_x = randint(*blur_range)
        sigma_y = randint(*blur_range)
        ksize_x = 2 * int(4 * sigma_x + 0.5) + 1
        ksize_y = 2 * int(4 * sigma_y + 0.5) + 1
        image = cv2.GaussianBlur(image, (ksize_x, ksize_y), sigmaX=sigma_x, sigmaY=sigma_y)
    elif algorithm == 'sinc':
        # Apply a 2D sinc filter to each channel of the image
        size = randint(*blur_range)
        x = np.linspace(-size // 2, size // 2, size)
        xx, yy = np.meshgrid(x, x)
        kernel = np.sinc(xx) * np.sinc(yy)
        kernel /= kernel.sum()
        if len(image.shape) == 3:
            # Split image into individual color channels
            channels = list(cv2.split(image))
            # Apply 2D sinc filter to each channel
            for i in range(len(channels)):
                channels[i] = signal.convolve2d(channels[i], kernel, mode='same', boundary='symm')
            # Merge channels back together
            image = cv2.merge(channels)
        else:
            # Apply 2D sinc filter to grayscale image
            image = signal.convolve2d(image, kernel, mode='same', boundary='symm')
    return image

def apply_noise(image):
    # Choose noise algorithm
    if noise_randomize:
        algorithm = choice(noise_algorithms)
    else:
        algorithm = noise_algorithms[0]

    # Apply noise with chosen algorithm
    if algorithm == 'uniform':
        intensity = randint(*noise_range)
        noise = np.random.uniform(-intensity, intensity, image.shape)
        image = cv2.add(image, noise.astype(image.dtype))
    elif algorithm == 'gaussian':
        intensity = randint(*noise_range)
        noise = np.random.normal(0, intensity, image.shape)
        image = cv2.add(image, noise.astype(image.dtype))
    elif algorithm == 'color':
        noise = np.zeros_like(image)
        m = (0, 0, 0)
        s = (randint(*noise_range), randint(*noise_range), randint(*noise_range))
        cv2.randn(noise, m, s)
        image += noise
    elif algorithm == 'gray':
        gray_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        m = (0,)
        s = (randint(*noise_range),)
        cv2.randn(gray_noise, m, s)
        gray_noise = gray_noise[..., None]
        image += gray_noise
    return image

def apply_compression(image):
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
    elif algorithm == 'webp':
        quality = randint(*webp_quality_range)
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        result, encimg = cv2.imencode('.webp', image, encode_param)
        image = cv2.imdecode(encimg, 1).copy()
    elif algorithm in ['h264', 'h265', 'mpeg', 'mpeg2']:
        # Convert image to video format
        height, width, _ = image.shape
        codec = algorithm
        container = 'mp4'
        if algorithm == 'mpeg':
            codec = 'mpeg1video'
            container = 'mpeg'
        elif algorithm == 'mpeg2':
            codec = 'mpeg2video'
            container = 'mpeg'
        
        # Get CRF level from config
        crf_level = config.getint('compression', 'crf_level')
        
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}')
            .output(f'temp.{container}', vcodec=codec, crf=crf_level)
            .global_args('-loglevel', 'quiet')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        process.stdin.write(image.tobytes())
        process.stdin.close()
        process.wait()

        # Read compressed video back into image format
        out, _ = (
            ffmpeg
            .input(f'temp.{container}')
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .global_args('-loglevel', 'quiet')
            .run(capture_stdout=True)
        )
        #image = np.frombuffer(out, np.uint8).reshape([height, width, 3]).copy()
    return image

def apply_scale(image):
    # Get output size factor from config
    size_factor = config.getfloat('scale', 'size_factor')

    # Calculate new size
    h, w = image.shape[:2]
    new_h = int(h * size_factor)
    new_w = int(w * size_factor)

    # Choose scale algorithm
    if scale_randomize:
        algorithm = choice(scale_algorithms)
    else:
        algorithm = scale_algorithms[0]

    if algorithm == 'down_up':
        scale_factor = np.random.uniform(*scale_range)
        image = cv2.resize(image, (int(w*scale_factor), int(h*scale_factor)))
        image = cv2.resize(image, (new_w, new_h))
    elif algorithm == 'bicubic':
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif algorithm == 'bilinear':
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    elif algorithm == 'box':
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    elif algorithm == 'nearest':
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    elif algorithm == 'lanczos':
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return image

def process_image(image_path):
    image = cv2.imread(image_path)

    degradation_order = degradations.copy()
    if degradations_randomize:
        shuffle(degradation_order)
    for degradation in degradation_order:
        if degradation == 'blur':
            image = apply_blur(image)
        elif degradation == 'noise':
            image = apply_noise(image)
        elif degradation == 'compression':
            image = apply_compression(image)
        elif degradation == 'scale':
            image = apply_scale(image)

    output_path = os.path.join(output_folder, os.path.relpath(image_path, input_folder))
    output_path = os.path.splitext(output_path)[0] + '.' + output_format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

# Process images recursively
image_paths = []
for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        image_paths.append(os.path.join(subdir, file))

for image_path in tqdm(image_paths):
    process_image(image_path)
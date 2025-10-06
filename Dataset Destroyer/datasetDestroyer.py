import configparser
import os
import cv2
import numpy as np
import ffmpeg
from random import random, randint, choice, shuffle, uniform
import concurrent.futures
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from chainner_ext import DiffusionAlgorithm, UniformQuantization, error_diffusion_dither, resize, ResizeFilter
import logging

logging.basicConfig(level=logging.DEBUG)

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

# Helper function to parse config values
def get_config(section, key, parser=str, default=None):
    """Parse config values with type conversion"""
    try:
        if parser == bool:
            return config.getboolean(section, key)
        elif parser == float:
            return config.getfloat(section, key)
        elif parser == int:
            return config.getint(section, key)
        elif parser == 'int_tuple':
            return tuple(map(int, config.get(section, key).split(',')))
        elif parser == 'float_tuple':
            return tuple(map(float, config.get(section, key).split(',')))
        elif parser == 'list':
            return config.get(section, key).split(',')
        else:
            return config.get(section, key)
    except:
        return default

# Main config
input_folder = get_config('main', 'input_folder')
output_folder = get_config('main', 'output_folder')
output_format = get_config('main', 'output_format')
degradations = get_config('main', 'degradations', 'list')
degradations_randomize = get_config('main', 'randomize', bool)
print_to_image = get_config('main', 'print', bool)
print_to_textfile = get_config('main', 'textfile', bool)
path_to_textfile = get_config('main', 'textfile_path')

# Blur config
blur_config = {
    'algorithms': get_config('blur', 'algorithms', 'list'),
    'randomize': get_config('blur', 'randomize', bool),
    'range': get_config('blur', 'range', 'int_tuple'),
    'scale_factor': get_config('blur', 'scale_factor', float)
}

# Noise config
noise_config = {
    'algorithms': get_config('noise', 'algorithms', 'list'),
    'randomize': get_config('noise', 'randomize', bool),
    'range': get_config('noise', 'range', 'int_tuple'),
    'scale_factor': get_config('noise', 'scale_factor', float),
    'sp_range': get_config('noise', 'sp_range', 'int_tuple'),
    'sp_scale_factor': get_config('noise', 'sp_scale_factor', float)
}

# Chroma config
chroma_config = {
    'algorithms': get_config('chroma', 'algorithms', 'list'),
    'randomize': get_config('chroma', 'randomize', bool),
    'horizontal_range': get_config('chroma', 'horizontal_range', 'int_tuple'),
    'vertical_range': get_config('chroma', 'vertical_range', 'int_tuple'),
    'scale_factor': get_config('chroma', 'scale_factor', float)
}

# Compression config
compression_config = {
    'algorithms': get_config('compression', 'algorithms', 'list'),
    'randomize': get_config('compression', 'randomize', bool),
    'jpeg_quality_range': get_config('compression', 'jpeg_quality_range', 'int_tuple'),
    'webp_quality_range': get_config('compression', 'webp_quality_range', 'int_tuple'),
    'h264_crf_level_range': get_config('compression', 'h264_crf_level_range', 'int_tuple'),
    'hevc_crf_level_range': get_config('compression', 'hevc_crf_level_range', 'int_tuple'),
    'vp9_crf_level_range': get_config('compression', 'vp9_crf_level_range', 'int_tuple'),
    'mpeg_qscale_range': get_config('compression', 'mpeg_qscale_range', 'int_tuple'),
    'mpeg2_qscale_range': get_config('compression', 'mpeg2_qscale_range', 'int_tuple')
}

# Scale config
scale_config = {
    'size_factor': get_config('scale', 'size_factor', float),
    'algorithms': get_config('scale', 'algorithms', 'list'),
    'down_up_algorithms': get_config('scale', 'down_up_algorithms', 'list'),
    'randomize': get_config('scale', 'randomize', bool),
    'range': get_config('scale', 'range', 'float_tuple')
}

# Quantization config
quantization_config = {
    'algorithms': get_config('quantization', 'algorithms', 'list'),
    'randomize': get_config('quantization', 'randomize', bool),
    'range': get_config('quantization', 'range', 'int_tuple')
}

# Unsharp mask config
unsharp_config = {
    'radius_range': get_config('unsharp_mask', 'radius_range', 'float_tuple'),
    'percent_range': get_config('unsharp_mask', 'percent_range', 'float_tuple'),
    'threshold_range': get_config('unsharp_mask', 'threshold_range', 'int_tuple')
}

# Likelihood config
likelihood_config = {
    'blur': get_config('likelihood', 'blur', float, 0.3),
    'noise': get_config('likelihood', 'noise', float, 0.3),
    'compression': get_config('likelihood', 'compression', float, 0.2),
    'scale': get_config('likelihood', 'scale', float, 0.1),
    'quantization': get_config('likelihood', 'quantization', float, 0.1),
    'unsharp_mask': get_config('likelihood', 'unsharp_mask', float, 0.1),
    'chroma': get_config('likelihood', 'chroma', float, 0.3)
}

def print_text_to_image(image, text, order):
    h, w = image.shape[:2]
    font_scale = min(w, h) / 1000
    font_thickness = max(1, int(font_scale * 2))
    
    # Break long text into multiple lines
    max_line_length = 40
    lines = []
    while len(text) > max_line_length:
        split_index = text[:max_line_length].rfind(' ')
        if split_index == -1:
            split_index = max_line_length
        lines.append(text[:split_index])
        text = text[split_index:].strip()
    lines.append(text)
    
    color = (0, 0, 255)  # Red in BGR
    text_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0] for line in lines]
    text_heights = [size[1] for size in text_sizes]
    
    x = 10
    y = int(order * text_heights[0] * 1.5) + 10
    
    for i, line in enumerate(lines):
        current_y = y + i * int(text_heights[0] * 1.5)
        cv2.putText(image, f"{order}. {line}" if i == 0 else line, 
                    (x, current_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, font_thickness, cv2.LINE_AA)
    
    return image

def print_text_to_textfile(file_name, text_to_append):
    with open(file_name, "a+") as file_object:
        file_object.seek(0)
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        file_object.write(text_to_append)

def select_algorithm(algorithms, randomize):
    """Helper to select algorithm based on randomize flag"""
    return choice(algorithms) if randomize else algorithms[0]

def ensure_odd(value):
    """Ensure value is odd"""
    return value if value % 2 == 1 else value + 1

def apply_blur(image):
    cfg = blur_config
    algorithm = select_algorithm(cfg['algorithms'], cfg['randomize'])
    
    # Normalize to [0, 1]
    image = image.astype(float) / 255
    
    if algorithm == 'average':
        ksize = ensure_odd(int(randint(*cfg['range']) * cfg['scale_factor']))
        image = cv2.blur(image, (ksize, ksize))
        text = f"{algorithm} ksize={ksize}"
    elif algorithm == 'gaussian':
        ksize = ensure_odd(int(randint(*cfg['range']) * cfg['scale_factor']))
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        text = f"{algorithm} ksize={ksize}"
    elif algorithm == 'anisotropic':
        sigma_x = randint(*cfg['range']) * cfg['scale_factor']
        sigma_y = randint(*cfg['range']) * cfg['scale_factor']
        angle = np.deg2rad(uniform(0, 360))
        
        kernel_size = max(2 * int(4 * max(sigma_x, sigma_y) + 0.5) + 1, 3)
        y, x = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, -kernel_size//2 + 1:kernel_size//2 + 1]
        rotx = x * np.cos(angle) - y * np.sin(angle)
        roty = x * np.sin(angle) + y * np.cos(angle)
        kernel = np.exp(-(rotx**2/(2*sigma_x**2) + roty**2/(2*sigma_y**2)))
        kernel /= np.sum(kernel)
        image = cv2.filter2D(image, -1, kernel)
        
        text = f"{algorithm} sigma_x={sigma_x} sigma_y={sigma_y} angle={np.rad2deg(angle)}"
    
    return (image * 255).astype(np.uint8), text

def apply_noise(image):
    cfg = noise_config
    algorithm = select_algorithm(cfg['algorithms'], cfg['randomize'])
    
    # Normalize to [0, 1]
    image = image.astype(float) / 255
    
    if algorithm == 'uniform':
        intensity = randint(*cfg['range']) * cfg['scale_factor']
        noise = np.random.uniform(-intensity, intensity, image.shape)
        image += noise
        text = f"{algorithm} intensity={intensity}"
    elif algorithm == 'gaussian':
        var = randint(*cfg['range']) * cfg['scale_factor']
        noise = np.random.normal(0, var**0.5, image.shape)
        image += noise
        text = f"{algorithm} variance={var}"
    elif algorithm == 'color':
        noise = np.zeros_like(image)
        s = tuple(randint(*cfg['range']) * cfg['scale_factor'] for _ in range(3))
        cv2.randn(noise, (0, 0, 0), s)
        image += noise
        text = f"{algorithm} s={s}"
    elif algorithm == 'gray':
        gray_noise = np.zeros((image.shape[0], image.shape[1]))
        s = (randint(*cfg['range']) * cfg['scale_factor'],)
        cv2.randn(gray_noise, (0,), s)
        for i in range(image.shape[2]):
            image[..., i] += gray_noise
        text = f"{algorithm} s={s}"
    elif algorithm == 'salt-and-pepper':
        intensity = randint(*cfg['sp_range']) * cfg['sp_scale_factor']
        num_pepper = np.ceil(intensity * image.size * 0.25)
        x_pepper = np.random.randint(0, image.shape[1], int(num_pepper))
        y_pepper = np.random.randint(0, image.shape[0], int(num_pepper))
        image[y_pepper, x_pepper] = 0
        
        num_salt = np.ceil(intensity * image.size * 0.5)
        x_salt = np.random.randint(0, image.shape[1], int(num_salt))
        y_salt = np.random.randint(0, image.shape[0], int(num_salt))
        image[y_salt, x_salt] = 1
        text = f"{algorithm} intensity={intensity}"
    
    return (np.clip(image, 0, 1) * 255).astype(np.uint8), text

def apply_chroma(image):
    cfg = chroma_config
    algorithm = select_algorithm(cfg['algorithms'], cfg['randomize'])
    
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    Y, U, V = cv2.split(yuv_image)
    
    if algorithm == 'gaussian':
        horizontal_ksize = ensure_odd(randint(*cfg['horizontal_range']))
        vertical_ksize = ensure_odd(randint(*cfg['vertical_range']))
        
        blurred_U = cv2.GaussianBlur(U, (horizontal_ksize, vertical_ksize), 0)
        blurred_V = cv2.GaussianBlur(V, (horizontal_ksize, vertical_ksize), 0)
        
        blurred_yuv_image = cv2.merge([Y, blurred_U, blurred_V])
        image = cv2.cvtColor(blurred_yuv_image, cv2.COLOR_YUV2RGB)
        
        text = f"{algorithm} horizontal_ksize={horizontal_ksize} vertical_ksize={vertical_ksize}"
    
    return image, text

def apply_quantization(image):
    cfg = quantization_config
    algorithm = select_algorithm(cfg['algorithms'], cfg['randomize'])
    
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
    
    if algorithm not in algorithm_mapping:
        raise ValueError(f"Unsupported quantization algorithm: {algorithm}")
    
    colors_per_channel = randint(*cfg['range'])
    quant = UniformQuantization(colors_per_channel=colors_per_channel)
    image_np = np.array(image).astype(np.float32) / 255.0
    
    for i in range(image_np.shape[2]):
        dithered_channel = error_diffusion_dither(image_np[..., i], quant, algorithm_mapping[algorithm])
        if len(dithered_channel.shape) == 3:
            dithered_channel = dithered_channel.squeeze(-1)
        image_np[..., i] = dithered_channel
    
    dithered_image_np = np.round(image_np * 255).astype(np.uint8)
    
    text = f"{algorithm} colors_per_channel={colors_per_channel}"
    return np.array(Image.fromarray(dithered_image_np)), text

def apply_unsharp_mask(image):
    cfg = unsharp_config
    radius = np.random.uniform(*cfg['radius_range'])
    percent = np.random.uniform(*cfg['percent_range'])
    threshold = np.random.randint(*cfg['threshold_range'])
    
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharpened = cv2.addWeighted(image, 1.0 + percent, blurred, -percent, threshold)
    image = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    text = f"unsharp_mask radius={radius} percent={percent} threshold={threshold}"
    return image, text

def apply_compression(image):
    cfg = compression_config
    algorithm = select_algorithm(cfg['algorithms'], cfg['randomize'])
    
    if algorithm == 'jpeg':
        quality = randint(*cfg['jpeg_quality_range'])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        image = cv2.imdecode(encimg, 1).copy()
        text = f"{algorithm} quality={quality}"
    elif algorithm == 'webp':
        quality = randint(*cfg['webp_quality_range'])
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        result, encimg = cv2.imencode('.webp', image, encode_param)
        image = cv2.imdecode(encimg, 1).copy()
        text = f"{algorithm} quality={quality}"
    elif algorithm in ['h264', 'hevc', 'mpeg', 'mpeg2', 'vp9']:
        height, width, _ = image.shape
        codec = algorithm
        container = 'mpeg'
        input_args = {}
        
        codec_configs = {
            'mpeg': ('mpeg1video', 'mpeg', {'framerate': '25'}, 
                     {'qscale:v': str(randint(*cfg['mpeg_qscale_range'])), 'g': '1', 'bf': '0'}),
            'mpeg2': ('mpeg2video', 'mpeg', {'framerate': '25'}, 
                      {'qscale:v': str(randint(*cfg['mpeg2_qscale_range'])), 'g': '1', 'bf': '0'}),
            'h264': ('h264', 'mpeg', {}, {'crf': randint(*cfg['h264_crf_level_range'])}),
            'hevc': ('hevc', 'mpeg', {}, {'crf': randint(*cfg['hevc_crf_level_range']), 'x265-params': 'log-level=0'}),
            'vp9': ('libvpx-vp9', 'webm', {}, {'crf': str(randint(*cfg['vp9_crf_level_range'])), 'b:v': '0', 'cpu-used': '5'})
        }
        
        codec, container, input_args, output_args = codec_configs[algorithm]
        
        process1 = None
        process2 = None
        
        try:
            process1 = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', **input_args)
                .output('pipe:', format=container, vcodec=codec, **output_args)
                .global_args('-loglevel', 'fatal')
                .global_args('-max_muxing_queue_size', '300000')
                .run_async(pipe_stdin=True, pipe_stdout=True)
            )
            
            process1.stdin.write(image.tobytes())
            process1.stdin.close()
            compressed_output = process1.stdout.read()
            process1.wait(timeout=10)
            
            if process1.returncode != 0:
                raise RuntimeError(f"FFmpeg encoding failed with return code {process1.returncode}")
            
            process2 = (
                ffmpeg
                .input('pipe:', format=container)
                .output('pipe:', format='rawvideo', pix_fmt='bgr24')
                .global_args('-loglevel', 'fatal')
                .run_async(pipe_stdin=True, pipe_stdout=True)
            )
            
            process2.stdin.write(compressed_output)
            process2.stdin.close()
            out = process2.stdout.read()
            process2.wait(timeout=10)
            
            if process2.returncode != 0:
                raise RuntimeError(f"FFmpeg decoding failed with return code {process2.returncode}")
            
            image = np.frombuffer(out, np.uint8)[:(height * width * 3)].reshape([height, width, 3]).copy()
        except Exception as e:
            logging.error(f"FFmpeg processing failed: {str(e)}")
            for p in [process1, process2]:
                try:
                    if p and p.poll() is None:
                        p.kill()
                except Exception:
                    logging.exception("Error cleaning up processes")
            raise
        
        first_arg = list(output_args.items())[0]
        text = f"{algorithm} {first_arg[0]}={first_arg[1]}"
    
    return image, text

def apply_scale(image):
    cfg = scale_config
    image = np.float32(image) / 255.0
    
    h, w = image.shape[:2]
    new_h = int(h * cfg['size_factor'])
    new_w = int(w * cfg['size_factor'])
    
    algorithm = select_algorithm(cfg['algorithms'], cfg['randomize'])
    
    interpolation_map = {
        'nearest': ResizeFilter.Nearest, 'box': ResizeFilter.Box, 'hermite': ResizeFilter.Hermite,
        'hamming': ResizeFilter.Hamming, 'linear': ResizeFilter.Linear, 'hann': ResizeFilter.Hann,
        'lagrange': ResizeFilter.Lagrange, 'cubic_catrom': ResizeFilter.CubicCatrom,
        'cubic_mitchell': ResizeFilter.CubicMitchell, 'cubic_bspline': ResizeFilter.CubicBSpline,
        'lanczos': ResizeFilter.Lanczos, 'gauss': ResizeFilter.Gauss
    }
    
    if algorithm == 'down_up':
        algorithm1 = select_algorithm(cfg['down_up_algorithms'], cfg['randomize'])
        algorithm2 = select_algorithm(cfg['down_up_algorithms'], cfg['randomize']) if cfg['randomize'] else cfg['down_up_algorithms'][-1]
        scale_factor = np.random.uniform(*cfg['range'])
        
        use_gamma1 = algorithm1 != 'nearest'
        use_gamma2 = algorithm2 != 'nearest'
        image = resize(image, (int(w * scale_factor), int(h * scale_factor)), interpolation_map[algorithm1], gamma_correction=use_gamma1)
        image = resize(image, (new_w, new_h), interpolation_map[algorithm2], gamma_correction=use_gamma2)
        text = f"{algorithm} scale1factor={scale_factor:.2f} scale1algorithm={algorithm1} scale2factor={cfg['size_factor']/scale_factor:.2f} scale2algorithm={algorithm2}"
    else:
        use_gamma = algorithm != 'nearest'
        image = resize(image, (new_w, new_h), interpolation_map[algorithm], gamma_correction=use_gamma)
        text = f"{algorithm} size factor={cfg['size_factor']}"
    
    return (image * 255).astype(np.uint8), text

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return
    
    # Build degradation order
    if degradations_randomize:
        degradation_order = [deg for deg in degradations if random() < likelihood_config.get(deg, 0)]
        shuffle(degradation_order)
    else:
        degradation_order = degradations.copy()
    
    degradation_funcs = {
        'blur': apply_blur,
        'noise': apply_noise,
        'chroma': apply_chroma,
        'compression': apply_compression,
        'scale': apply_scale,
        'quantization': apply_quantization,
        'unsharp_mask': apply_unsharp_mask
    }
    
    all_text = []
    for degradation in degradation_order:
        image, text = degradation_funcs[degradation](image)
        all_text.append(f"{degradation}: {text}")
    
    if print_to_image:
        for order, text in enumerate(all_text, 1):
            image = print_text_to_image(image, text, order)
    
    # Save image
    output_path = os.path.join(output_folder, os.path.relpath(image_path, input_folder))
    output_path = os.path.splitext(output_path)[0] + '.' + output_format
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    
    if print_to_textfile:
        print_text_to_textfile(path_to_textfile + "/applied_degradations.txt", os.path.basename(output_path) + ' - ' + ', '.join(all_text))

# Collect image paths efficiently
image_paths = list(Path(input_folder).rglob('*.*'))
image_paths = [str(p) for p in image_paths if p.is_file()]

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_image, image_path) for image_path in image_paths}
        kwargs = {'total': len(futures), 'unit': 'it', 'unit_scale': True, 'leave': True}
        
        try:
            for f in tqdm(concurrent.futures.as_completed(futures), **kwargs):
                try:
                    f.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
        except KeyboardInterrupt:
            print("Interrupted by user, terminating processes...")
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()
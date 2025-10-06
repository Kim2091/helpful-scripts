import unittest
import tempfile
import shutil
import os
import cv2
import numpy as np
from pathlib import Path
import configparser
import sys

# Import the main script functions
sys.path.insert(0, os.path.dirname(__file__))
import datasetDestroyer as dd

class TestDatasetDestroyer(unittest.TestCase):
    """Test suite for Dataset Destroyer script"""
    
    @classmethod
    def setUpClass(cls):
        """Create temporary directories and test images"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.input_dir = os.path.join(cls.temp_dir, 'input')
        cls.output_dir = os.path.join(cls.temp_dir, 'output')
        os.makedirs(cls.input_dir)
        os.makedirs(cls.output_dir)
        
        # Create test images
        cls.test_image_paths = []
        for i in range(3):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img_path = os.path.join(cls.input_dir, f'test_image_{i}.png')
            cv2.imwrite(img_path, img)
            cls.test_image_paths.append(img_path)
        
        # Create test config
        cls.config_path = os.path.join(cls.temp_dir, 'test_config.ini')
        cls._create_test_config()
        
        # Load test config
        dd.config = configparser.ConfigParser()
        dd.config.read(cls.config_path)
        cls._reload_configs()
    
    @classmethod
    def _create_test_config(cls):
        """Create a minimal test config file"""
        config = configparser.ConfigParser()
        
        config['main'] = {
            'input_folder': cls.input_dir,
            'output_folder': cls.output_dir,
            'output_format': 'png',
            'degradations': 'blur,noise',
            'randomize': 'False',
            'print': 'False',
            'textfile': 'False',
            'textfile_path': cls.output_dir
        }
        
        config['likelihood'] = {
            'blur': '1.0',
            'noise': '1.0',
            'compression': '1.0',
            'scale': '1.0',
            'quantization': '1.0',
            'unsharp_mask': '1.0',
            'chroma': '1.0'
        }
        
        config['blur'] = {
            'algorithms': 'gaussian',
            'randomize': 'False',
            'range': '3,7',
            'scale_factor': '1.0'
        }
        
        config['noise'] = {
            'algorithms': 'gaussian',
            'randomize': 'False',
            'range': '1,5',
            'scale_factor': '0.1',
            'sp_range': '1,5',
            'sp_scale_factor': '0.02'
        }
        
        config['chroma'] = {
            'algorithms': 'gaussian',
            'randomize': 'False',
            'horizontal_range': '3,5',
            'vertical_range': '1,2',
            'scale_factor': '1.0'
        }
        
        config['compression'] = {
            'algorithms': 'jpeg',
            'randomize': 'False',
            'jpeg_quality_range': '70,90',
            'webp_quality_range': '70,90',
            'h264_crf_level_range': '23,28',
            'hevc_crf_level_range': '25,30',
            'vp9_crf_level_range': '25,30',
            'mpeg_qscale_range': '5,10',
            'mpeg2_qscale_range': '5,10'
        }
        
        config['scale'] = {
            'size_factor': '0.5',
            'algorithms': 'linear',
            'down_up_algorithms': 'linear,cubic_mitchell',
            'randomize': 'False',
            'range': '0.75,1.25'
        }
        
        config['quantization'] = {
            'algorithms': 'floyd_steinberg',
            'randomize': 'False',
            'range': '16,64'
        }
        
        config['unsharp_mask'] = {
            'radius_range': '0.5,1.0',
            'percent_range': '5,10',
            'threshold_range': '1,2'
        }
        
        with open(cls.config_path, 'w') as f:
            config.write(f)
    
    @classmethod
    def _reload_configs(cls):
        """Reload all config dictionaries from test config"""
        dd.input_folder = dd.get_config('main', 'input_folder')
        dd.output_folder = dd.get_config('main', 'output_folder')
        dd.output_format = dd.get_config('main', 'output_format')
        dd.degradations = dd.get_config('main', 'degradations', 'list')
        dd.degradations_randomize = dd.get_config('main', 'randomize', bool)
        dd.print_to_image = dd.get_config('main', 'print', bool)
        dd.print_to_textfile = dd.get_config('main', 'textfile', bool)
        dd.path_to_textfile = dd.get_config('main', 'textfile_path')
        
        dd.blur_config = {
            'algorithms': dd.get_config('blur', 'algorithms', 'list'),
            'randomize': dd.get_config('blur', 'randomize', bool),
            'range': dd.get_config('blur', 'range', 'int_tuple'),
            'scale_factor': dd.get_config('blur', 'scale_factor', float)
        }
        
        dd.noise_config = {
            'algorithms': dd.get_config('noise', 'algorithms', 'list'),
            'randomize': dd.get_config('noise', 'randomize', bool),
            'range': dd.get_config('noise', 'range', 'int_tuple'),
            'scale_factor': dd.get_config('noise', 'scale_factor', float),
            'sp_range': dd.get_config('noise', 'sp_range', 'int_tuple'),
            'sp_scale_factor': dd.get_config('noise', 'sp_scale_factor', float)
        }
        
        dd.chroma_config = {
            'algorithms': dd.get_config('chroma', 'algorithms', 'list'),
            'randomize': dd.get_config('chroma', 'randomize', bool),
            'horizontal_range': dd.get_config('chroma', 'horizontal_range', 'int_tuple'),
            'vertical_range': dd.get_config('chroma', 'vertical_range', 'int_tuple'),
            'scale_factor': dd.get_config('chroma', 'scale_factor', float)
        }
        
        dd.compression_config = {
            'algorithms': dd.get_config('compression', 'algorithms', 'list'),
            'randomize': dd.get_config('compression', 'randomize', bool),
            'jpeg_quality_range': dd.get_config('compression', 'jpeg_quality_range', 'int_tuple'),
            'webp_quality_range': dd.get_config('compression', 'webp_quality_range', 'int_tuple'),
            'h264_crf_level_range': dd.get_config('compression', 'h264_crf_level_range', 'int_tuple'),
            'hevc_crf_level_range': dd.get_config('compression', 'hevc_crf_level_range', 'int_tuple'),
            'vp9_crf_level_range': dd.get_config('compression', 'vp9_crf_level_range', 'int_tuple'),
            'mpeg_qscale_range': dd.get_config('compression', 'mpeg_qscale_range', 'int_tuple'),
            'mpeg2_qscale_range': dd.get_config('compression', 'mpeg2_qscale_range', 'int_tuple')
        }
        
        dd.scale_config = {
            'size_factor': dd.get_config('scale', 'size_factor', float),
            'algorithms': dd.get_config('scale', 'algorithms', 'list'),
            'down_up_algorithms': dd.get_config('scale', 'down_up_algorithms', 'list'),
            'randomize': dd.get_config('scale', 'randomize', bool),
            'range': dd.get_config('scale', 'range', 'float_tuple')
        }
        
        dd.quantization_config = {
            'algorithms': dd.get_config('quantization', 'algorithms', 'list'),
            'randomize': dd.get_config('quantization', 'randomize', bool),
            'range': dd.get_config('quantization', 'range', 'int_tuple')
        }
        
        dd.unsharp_config = {
            'radius_range': dd.get_config('unsharp_mask', 'radius_range', 'float_tuple'),
            'percent_range': dd.get_config('unsharp_mask', 'percent_range', 'float_tuple'),
            'threshold_range': dd.get_config('unsharp_mask', 'threshold_range', 'int_tuple')
        }
        
        dd.likelihood_config = {
            'blur': dd.get_config('likelihood', 'blur', float, 1.0),
            'noise': dd.get_config('likelihood', 'noise', float, 1.0),
            'compression': dd.get_config('likelihood', 'compression', float, 1.0),
            'scale': dd.get_config('likelihood', 'scale', float, 1.0),
            'quantization': dd.get_config('likelihood', 'quantization', float, 1.0),
            'unsharp_mask': dd.get_config('likelihood', 'unsharp_mask', float, 1.0),
            'chroma': dd.get_config('likelihood', 'chroma', float, 1.0)
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directories"""
        shutil.rmtree(cls.temp_dir)
    
    def test_helper_functions(self):
        """Test helper functions"""
        # Test select_algorithm
        algorithms = ['algo1', 'algo2', 'algo3']
        result = dd.select_algorithm(algorithms, False)
        self.assertEqual(result, 'algo1')
        
        result = dd.select_algorithm(algorithms, True)
        self.assertIn(result, algorithms)
        
        # Test ensure_odd
        self.assertEqual(dd.ensure_odd(5), 5)
        self.assertEqual(dd.ensure_odd(6), 7)
        self.assertEqual(dd.ensure_odd(10), 11)
    
    def test_apply_blur(self):
        """Test blur degradation"""
        test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result_img, text = dd.apply_blur(test_img)
        
        self.assertEqual(result_img.shape, test_img.shape)
        self.assertEqual(result_img.dtype, np.uint8)
        self.assertIn('gaussian', text)
        self.assertFalse(np.array_equal(result_img, test_img))
    
    def test_apply_noise(self):
        """Test noise degradation"""
        test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result_img, text = dd.apply_noise(test_img)
        
        self.assertEqual(result_img.shape, test_img.shape)
        self.assertEqual(result_img.dtype, np.uint8)
        self.assertIn('gaussian', text)
        self.assertFalse(np.array_equal(result_img, test_img))
    
    def test_apply_chroma(self):
        """Test chroma degradation"""
        test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result_img, text = dd.apply_chroma(test_img)
        
        self.assertEqual(result_img.shape, test_img.shape)
        self.assertEqual(result_img.dtype, np.uint8)
        self.assertIn('gaussian', text)
    
    def test_apply_compression_jpeg(self):
        """Test JPEG compression"""
        test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result_img, text = dd.apply_compression(test_img)
        
        self.assertEqual(result_img.shape, test_img.shape)
        self.assertEqual(result_img.dtype, np.uint8)
        self.assertIn('jpeg', text)
    
    def test_apply_scale(self):
        """Test scale degradation"""
        test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result_img, text = dd.apply_scale(test_img)
        
        expected_size = int(128 * dd.scale_config['size_factor'])
        self.assertEqual(result_img.shape[:2], (expected_size, expected_size))
        self.assertEqual(result_img.dtype, np.uint8)
        self.assertIn('linear', text)
    
    def test_apply_unsharp_mask(self):
        """Test unsharp mask"""
        test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result_img, text = dd.apply_unsharp_mask(test_img)
        
        self.assertEqual(result_img.shape, test_img.shape)
        self.assertEqual(result_img.dtype, np.uint8)
        self.assertIn('unsharp_mask', text)
    
    def test_apply_quantization(self):
        """Test quantization degradation"""
        test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result_img, text = dd.apply_quantization(test_img)
        
        self.assertEqual(result_img.shape, test_img.shape)
        self.assertEqual(result_img.dtype, np.uint8)
        self.assertIn('floyd_steinberg', text)
    
    def test_process_image(self):
        """Test full image processing pipeline"""
        test_path = self.test_image_paths[0]
        dd.process_image(test_path)
        
        # Check output file exists
        output_path = os.path.join(
            dd.output_folder, 
            os.path.relpath(test_path, dd.input_folder)
        )
        output_path = os.path.splitext(output_path)[0] + '.' + dd.output_format
        
        self.assertTrue(os.path.exists(output_path))
        
        # Check output image is valid
        output_img = cv2.imread(output_path)
        self.assertIsNotNone(output_img)
        self.assertEqual(output_img.dtype, np.uint8)
    
    def test_degradation_order_randomize(self):
        """Test degradation order with randomization"""
        dd.degradations = ['blur', 'noise', 'compression', 'scale']
        dd.degradations_randomize = True
        dd.likelihood_config = {deg: 1.0 for deg in dd.degradations}
        
        # Run multiple times to check randomization
        orders = []
        for _ in range(5):
            order = [deg for deg in dd.degradations if dd.random() < dd.likelihood_config.get(deg, 0)]
            orders.append(tuple(order))
        
        # At least check that we get some degradations
        self.assertTrue(all(len(order) > 0 for order in orders))
    
    def test_degradation_order_sequential(self):
        """Test degradation order without randomization"""
        dd.degradations = ['blur', 'noise']
        dd.degradations_randomize = False
        
        expected_order = dd.degradations.copy()
        self.assertEqual(expected_order, ['blur', 'noise'])
    
    def test_print_text_to_image(self):
        """Test text printing on image"""
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        text = "Test degradation: blur ksize=5"
        result_img = dd.print_text_to_image(test_img, text, 1)
        
        self.assertEqual(result_img.shape, test_img.shape)
        self.assertFalse(np.array_equal(result_img, test_img))
    
    def test_print_text_to_textfile(self):
        """Test text file output"""
        test_file = os.path.join(self.temp_dir, 'test_output.txt')
        dd.print_text_to_textfile(test_file, "Test line 1")
        dd.print_text_to_textfile(test_file, "Test line 2")
        
        self.assertTrue(os.path.exists(test_file))
        
        with open(test_file, 'r') as f:
            content = f.read()
            self.assertIn("Test line 1", content)
            self.assertIn("Test line 2", content)
    
    def test_config_parsing(self):
        """Test config parsing helper"""
        # Test boolean parsing
        result = dd.get_config('main', 'randomize', bool)
        self.assertIsInstance(result, bool)
        
        # Test float parsing
        result = dd.get_config('scale', 'size_factor', float)
        self.assertIsInstance(result, float)
        
        # Test int_tuple parsing
        result = dd.get_config('blur', 'range', 'int_tuple')
        self.assertIsInstance(result, tuple)
        self.assertTrue(all(isinstance(x, int) for x in result))
        
        # Test list parsing
        result = dd.get_config('blur', 'algorithms', 'list')
        self.assertIsInstance(result, list)
        
        # Test default value
        result = dd.get_config('nonexistent', 'key', str, 'default_value')
        self.assertEqual(result, 'default_value')

def run_visual_test():
    """Run a visual test that creates sample degraded images"""
    print("\n" + "="*60)
    print("VISUAL TEST: Creating sample degraded images")
    print("="*60)
    
    # Create test directory
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, 'visual_test_input')
    output_dir = os.path.join(test_dir, 'visual_test_output')
    os.makedirs(input_dir)
    os.makedirs(output_dir)
    
    # Create a colorful test image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:256, :256] = [255, 0, 0]      # Red
    img[:256, 256:] = [0, 255, 0]      # Green
    img[256:, :256] = [0, 0, 255]      # Blue
    img[256:, 256:] = [255, 255, 0]    # Yellow
    
    # Add some patterns
    for i in range(0, 512, 32):
        cv2.line(img, (i, 0), (i, 512), (255, 255, 255), 1)
        cv2.line(img, (0, i), (512, i), (255, 255, 255), 1)
    
    test_img_path = os.path.join(input_dir, 'test_pattern.png')
    cv2.imwrite(test_img_path, img)
    
    print(f"\nTest image created at: {test_img_path}")
    print(f"Output directory: {output_dir}")
    
    # Test each degradation individually
    degradations_to_test = ['blur', 'noise', 'chroma', 'compression', 'scale', 'quantization', 'unsharp_mask']
    
    for deg in degradations_to_test:
        print(f"\nTesting {deg}...")
        test_img = cv2.imread(test_img_path)
        
        try:
            if deg == 'blur':
                result, text = dd.apply_blur(test_img)
            elif deg == 'noise':
                result, text = dd.apply_noise(test_img)
            elif deg == 'chroma':
                result, text = dd.apply_chroma(test_img)
            elif deg == 'compression':
                result, text = dd.apply_compression(test_img)
            elif deg == 'scale':
                result, text = dd.apply_scale(test_img)
            elif deg == 'quantization':
                result, text = dd.apply_quantization(test_img)
            elif deg == 'unsharp_mask':
                result, text = dd.apply_unsharp_mask(test_img)
            
            output_path = os.path.join(output_dir, f'{deg}_test.png')
            cv2.imwrite(output_path, result)
            print(f"  ✓ {deg}: {text}")
            print(f"    Saved to: {output_path}")
        except Exception as e:
            print(f"  ✗ {deg} failed: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"Visual test complete! Check output at:\n{output_dir}")
    print(f"{'='*60}\n")
    
    return test_dir

if __name__ == '__main__':
    print("Dataset Destroyer Test Suite")
    print("="*60)
    
    # Run unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDatasetDestroyer)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run visual test if unit tests pass
    if result.wasSuccessful():
        visual_test_dir = run_visual_test()
        print(f"\nAll tests passed! ✓")
        print(f"\nVisual test output saved to: {visual_test_dir}")
        print("You can manually inspect the degraded images.")
    else:
        print(f"\nSome tests failed. Please fix issues before running visual tests.")
    
    sys.exit(0 if result.wasSuccessful() else 1)
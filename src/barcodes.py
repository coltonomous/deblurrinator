import cv2 
import numpy as np
import os
import random

from barcode import UPCA # 1D barcode format used in https://www.math.mcgill.ca/rchoksi/pub/KL.pdf
from barcode.writer import ImageWriter # Needed to write to png image file
from pyzbar.pyzbar import decode # Detects and decodes barcodes in img
from scipy.optimize import minimize # L-BFGS used in minimzation of primal problem in paper
from scipy.signal import convolve2d
from scipy.signal import fftconvolve # Convolution function for deblurring
from scipy.stats import entropy # KL Divergence

# NOTE: The Windows version of pyzbar does not contain dlls for detecting rotated barcodes.
# Thus, in this environment images must be rotated manually such that barcodes are as
# close as possible to either 0 or 90(?) degrees relative to the window.

# TODO: Preprocess asset images (rotation, upscaling)
# TODO: Implement deblurring using method from paper
# TODO: Implement benchmarking between deblurring methods
# TODO: Bonus: Deblur images using dl model
# TODO: Bonus: Apply to image stream
# TODO: Output results to some presentable format


## Creates folder named *dir_name*
## param: (string) dir_name - folder name to create
## return: None
def create_resource_dir(dir_name):
    try:
        os.mkdir(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
        
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")
        
    except PermissionError:
        print(f"Permission denied: Unable to create '{dir_name}'.")
        
    except Exception as e:
        print(f"An error occurred: {e}")


## Creates *num* randomly generated test barcodes and saves to local subdir
## param: (int) num - number of barcode images to make
## return: (list) tuple of encoded barcode data and path of written barcode image
def generate_test_barcodes(num):
    # assert 1 <= num <= 20, "Passed num must be between 1 and 20."
    assert isinstance(num, int), "Passed num must be an integer."
    
    create_resource_dir("./test/source")
    barcode_paths = []
    
    for i in range(num):
        # Note: Needs to be string
        barcode_source = "{rand}".format(rand=random.randint(100000000000, 999999999999)) # 12 digits per UPCA def
        
        # TODO: Generalize to use more barcode types
        test_barcode = UPCA(barcode_source, writer=ImageWriter()) 
        write_path = test_barcode.save("./test/source/test_img_{iter}".format(iter=i),  options={"write_text": False, "quiet_zone": 0.5})
        
        barcode_paths.append((barcode_source, write_path))
        
    return barcode_paths
 

## Blurs *img_file* using randomly generated Gaussian blur kernel
## param: (list || string) image - loaded image matrix or filepath to image file
## param: (bool) save - toggle to write blurred image to file
## param: (int) bc_num - barcode sample number to append to saved image
## return: (image) blurred image, (string) path to saved image; empty if not saved
def apply_random_gaussian_blur(image, save=False, bc_num=0):
    
    # Allow passing in loaded cv image
    if isinstance(image, str):
        assert os.path.exists(image), "Image file not found at {path}".format(path=image)
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE) 
    
    rand_kernel = random.randrange(3, 25, 2)
    rand_sig = random.randint(1, 5)
    
    image = cv2.GaussianBlur(image, (rand_kernel, rand_kernel), rand_sig)
        
    write_path = ""
    
    if save:
        write_path = "./test/blur/gaussian_blur_{iter}.jpg".format(iter=bc_num)
        create_resource_dir("./test/blur")
        cv2.imwrite(write_path, image)
    
    return image, write_path


## Blurs *img_file* using supplied *kernel*
## param: (list || string) image - loaded image matrix or filepath to image file
## param: (list) kernel - 2D blurring kernel; unnormalized is fine
## param: (bool) save - toggle to write blurred image to file
## param: (int) bc_num - barcode sample number to append to saved image
## return: (image) blurred image, (string) path to saved image; empty if not saved
def apply_custom_blur(image, kernel, save=False, bc_num=0):
    # assert all([isinstance(i, numbers.Number) for i in kernel.flatten()]), "Kernel can only contain numbers"
    
    if isinstance(image, str):
        assert os.path.exists(image), "Image file not found at {path}".format(path=image)
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE) 
        
    np_kernel = np.array(kernel)
    kernel_normalized = np_kernel / np.sum(np_kernel) # Normalize the kernel
    blurred_img = cv2.filter2D(image, -1, kernel_normalized)
    write_path = ""
    
    if save:
        write_path = "./test/blur/custom_blur_{iter}.jpg".format(iter=bc_num)
        create_resource_dir("./test/blur")
        cv2.imwrite(write_path, blurred_img)
    
    return blurred_img, write_path

      
## Decodes barcode(s) contained within *img_file*
## param: (string) img_file - filepath to image file
## param: (bool) save - toggle to write detected barcodes in image to file
## param: (int) bc_num - barcode sample number to append to saved image
## return: (image) image with detected barcodes highlighted; unmodified source image if none found
def decode_barcode(image, save=False, bc_num=0):
    
    if isinstance(image, str):
        assert os.path.exists(image), "Image file not found at {path}".format(path=image)
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE) 
    
    detected_barcodes = decode(image) 
    write_path = ""
       
    if not detected_barcodes: 
        print("Barcode not detected or is blank/corrupted.") 
        
    else: 
        for barcode in detected_barcodes:   
            (x, y, w, h) = barcode.rect # Get barcode bounding box
              
            # Put rectangle around barcode in image; color is grayscale
            image = cv2.rectangle(image, (x, y), (x + w, y + h), 128, 2)
            
            if barcode.type and barcode.data:
                print("Detected barcode of type: {type} encoding: {data}".format(type=barcode.type, data=barcode.data))
                
    if save:
        write_path = "./{origin}/intermediate/detected_{iter}.jpg".format(iter=bc_num)
        create_resource_dir("./test/intermediate")
        cv2.imwrite(write_path, image)
            
    return image, write_path


def row_summation_threshold(image):
    if isinstance(image, str):
        assert os.path.exists(image), "Image file not found at {path}".format(path=image)
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE) 
        
    dens = np.sum(image, axis=0)
    mean = np.mean(dens)
    
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21)))
    for idx, val in enumerate(dens):
        if val < mean * 1.01:
            closed[:,idx] = 0

    _, thresh = cv2.threshold(closed, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    return thresh
        
    
def richardson_lucy_blind(image, num_iter=50):
    if isinstance(image, str):
        assert os.path.exists(image), "Image file not found at {path}".format(path=image)
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE) 
    
    kernel_size = 5
    psf = np.ones((kernel_size, kernel_size)) / kernel_size**2
    # psf = cv2.getGaussianKernel(kernel_size, 1)
    # psf = np.array([[.33, 0, 0], [.33, 0, 0], [.33, 0, 0]])
    psf_hat = psf[::-1, ::-1]
    psf_init = psf.copy()

    latent_est = image.copy()
    damping = 0.1

    # for _ in range(num_iter):
    #     est_conv = convolve2d(latent_est, psf, mode="same")
    #     relative_blur = image / est_conv
    #     error_est = convolve2d(relative_blur, psf_hat, mode="same")
    #     latent_est = latent_est * error_est
    
    for _ in range(num_iter):
        # Estimate image given PSF
        blurred_estimate = convolve2d(latent_est, psf, mode="same")
        relative_blur = blurred_image / blurred_estimate
        latent_est = latent_est * convolve2d(relative_blur, psf_hat, mode="same")

        # Estimate PSF given image
        blurred_estimate = convolve2d(latent_est, psf, mode="same")
        relative_blur = blurred_image / blurred_estimate
        psf = psf * convolve2d(latent_est[::-1, ::-1], relative_blur, mode="same")
        psf /= psf.sum()

        # Apply damping
        psf = (1 - damping) * psf + damping * psf_init

    return latent_est.astype("uint8")
    
    
def kl_based_blind(image, prior):
    if isinstance(image, str):
        assert os.path.exists(image), "Image file not found at {path}".format(path=image)
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    for i in range(1, int(min(image.shape) / 2) - 1): # Ensure kernel never gets larger than image; supports non-square images
        test_image = image.copy()
        kernel_size = 2 * i + 1
        kernel = cv2.getGaussianKernel(kernel_size, 1) # Start with normalized Gaussian kernel (1D)
        # kernel = np.ones(kernel_size) / kernel_size

        for _ in range(5):
            # Estimate blur kernel based on minimizing kl-divergence
            kernel = estimate_blur_kernel(test_image, kernel, prior) # ?
            
            # Convert new kernel to 2D and convolve image
            kernel_2d = np.outer(kernel, kernel)
            test_image = fftconvolve(test_image, kernel_2d, mode="same") #?
            
            print(kernel_2d)
            
            # Threshold image and decode to validate
            _, thresholded = cv2.threshold(test_image.astype("uint8"), 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            validated = decode(thresholded)
            
            # No need to continue if barcode is readable
            if validated:
                print("Good barcode")
                return thresholded
        break

    return image


def estimate_blur_kernel(blurred_image, kernel, prior):
    def objective_function(kernel):
        kernel_2d = np.outer(kernel, kernel)
        estimated_blurred_image = fftconvolve(blurred_image, kernel_2d, mode="same") 

        # Normalize the pixel values to create probability distributions
        observed, _ = np.histogram(estimated_blurred_image.flatten(), bins=256, range=(0, 256), density=True)

        # Calculate the KL Divergence
        print("objective call")
        print(kernel_2d)
        # print(entropy(observed, prior))
        
        return entropy(observed, observed)
    
    result = minimize(objective_function, kernel, method='L-BFGS-B')
    return result.x
    
    
def crop_to_barcodes(image):
    results = []
    detected_barcodes = decode(image)
    
    if detected_barcodes:
        for barcode in detected_barcodes:
            (x, y, w, h) = barcode.rect
            cropped = image[y:y+h, x:x+w]
            results.append(cropped)
    
    return results


def generate_emirical_prior():
    bernoulli_samples = []
   
    for file in os.listdir("./test/source"):
        sample = "./test/source/{file}".format(file=file)
        image = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
        image = crop_to_barcodes(image)[0] # Remove empty space around barcode
        
        # _, image = cv2.threshold(image.astype("uint8"), 128, 255, cv2.THRESH_BINARY_INV) # Threshold to remove intermediate values
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F) # Convert to binary
        image_flat = image.flatten()
        
        p_success = np.count_nonzero(image_flat, axis=0) / len(image_flat)
        bernoulli_samples.append(p_success)

    rho = sum(bernoulli_samples) / len(os.listdir("./test/source"))
    return rho
    

# def get_binomial_mu(rho):
#     mu = 1
#     for index, bar in enumerate(image[0]):
#         mu *= rho**(index + 1) * (1 - rho)**index
#     print(mu)    

    
if __name__ == "__main__":
    SAMPLE_SIZE = 1
    WRITE_INTERMEDIATE = False
    
    create_resource_dir("./test")
    samples = generate_test_barcodes(SAMPLE_SIZE)
    
    # Generate empirical prior # TODO: Increase sample size and store
    prior = generate_emirical_prior()
    print("success: {success}, other: {other}".format(success=prior, other=1-prior))
    
    for i in range(len(samples)):
        data = samples[i][0]
        path = samples[i][1]
        
        cv2.imshow("Original {data}".format(data=data), cv2.imread(path, cv2.IMREAD_GRAYSCALE)) # Base image
        
        blurred_image, _ = apply_random_gaussian_blur(path, WRITE_INTERMEDIATE, i)
        
        # linear_blur_kernel = np.rot90(np.diag([0.2] * 19)) # Custom linear blur kernel
        # blurred_image, _ = apply_custom_blur(blurred_image, linear_blur_kernel, WRITE_INTERMEDIATE, i)
        
        cv2.imshow("Blurred {data}".format(data=data), blurred_image)
        
        # Comparative tests
        # deblurred_image = row_summation_threshold("./assets/blind_0.jpg")
        # cv2.imshow("row_summation_threshold", deblurred_image)
        
        # deblurred_image = richardson_lucy_blind("./test/blur/gaussian_blur_0.jpg")
        # cv2.imshow("richardson_lucy_blind", deblurred_image)
        
        # deblurred_image = kl_based_blind(blurred_image, prior)
        # cv2.imshow("kl_based_blind", deblurred_image) # Deblurred image
        
        # decoded, _ = decode_barcode(deblurred_image, WRITE_INTERMEDIATE, i)
        # cv2.imshow("Decoded", decoded) # Final image
            
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    
    
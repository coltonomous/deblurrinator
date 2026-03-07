import cv2
import numpy as np
import os
import random

from barcode import UPCA
from barcode.writer import ImageWriter
from pyzbar.pyzbar import decode


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


if __name__ == "__main__":
    # Run the entropic blind deblurring demo from the paper's method
    from entropic_deblur import demo
    demo()

    
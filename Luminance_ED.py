from detector import *
import matplotlib.pyplot as plt

def Luminance_ED(img_name: str) -> np.ndarray:
    """
    Main method for Luminance Edge Detector

    Parameters
    ----------
    img_name : str
        Original image file name

    Returns
    -------
    np.ndarray
        Final edge detected image array

    """
    image = read_img(img_name)
    sobel_img = sobel_with_blur(image)
    theta_value = theta(sobel_img)
    non_max_suppress_img = non_max_suppression(sobel_img, theta_value)
    threshold_img = threshold(non_max_suppress_img)
    hys_img = hysteresis(threshold_img)
    final_img = bright(image, hys_img)
    return final_img

def trial():
    """
    Trial method to demonstrate edge detection using the included orig.jpg mri brain tumour image

    Returns
    -------
    None

    """
    ed_img = Luminance_ED("orig.jpg")
    plt.imshow(ed_img, 'gray')
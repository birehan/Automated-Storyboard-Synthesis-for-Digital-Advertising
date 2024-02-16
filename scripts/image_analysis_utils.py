import PIL
from PIL import Image
import webcolors
import pandas as pd
import cv2
import pytesseract
from logger import logger
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os
from rembg import remove 


def get_image_dimensions(image_path: str) -> tuple:
    """
    Get the width and height of an image.

    Args:
    - image_path (str): The path to the image file.

    Returns:
    - tuple: A tuple containing the width and height of the image.
    """
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image from {image_path}")

        # Retrieve the dimensions of the image
        height, width, _ = image.shape
        logger.info(f"Image dimensions retrieved successfully for {image_path}")
        return width, height
    except Exception as e:
        logger.error(f"An error occurred while getting image dimensions: {e}")
        return None, None

def extract_text_on_image(image_location: str) -> List[str]:
    """
    Extract text written on images using OCR (Optical Character Recognition).

    Args:
    - image_location (str): The path to the image file.

    Returns:
    - List[str]: A list of strings containing the extracted text from the image.
    """
    try:
        image = cv2.imread(image_location)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im2 = image.copy()
        string_array = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = im2[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped)
            string_array.append(text)
        clean_array = []
        for s in string_array:
            the_text = str(s).replace("\n", " ").replace("\x0c", "").replace("  ", " ").strip()
            if the_text != "":
                clean_array.append(the_text)
        logger.info("Text extracted from image successfully")
        return clean_array
    except Exception as e:
        logger.error(f"An unexpected error occurred while extracting text from image: {e}")
        return []

def closest_colour(requested_colour: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Find the closest color name from the CSS3 color names list for a given RGB color.

    Args:
    - requested_colour (Tuple[int, int, int]): The RGB color tuple for which the closest color name is to be found.

    Returns:
    - Tuple[int, int, int]: The RGB color tuple representing the closest color name.
    """
    try:
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = requested_colour
        return min_colours[min(min_colours.keys())]
    except Exception as e:
        logger.error(f"An unexpected error occurred while finding the closest color: {e}")


def top_colors(image: Image.Image, n: int) -> pd.Series:
    """
    Determine the dominant colors in an image.

    Args:
    - image (Image.Image): The input image.
    - n (int): The number of dominant colors to retrieve.

    Returns:
    - pd.Series: A pandas Series containing the dominant colors and their percentages in the image.
    """
    try:
        image = image.convert('RGB')
        image = image.resize((300, 300))
        detected_colors = []
        for x in range(image.width):
            for y in range(image.height):
                detected_colors.append(closest_colour(image.getpixel((x, y))))
        Series_Colors = pd.Series(detected_colors)
        output = Series_Colors.value_counts() / len(Series_Colors)
        logger.info("Dominant colors determined successfully")
        return output.head(n)
    except Exception as e:
        logger.error(f"An unexpected error occurred while determining dominant colors: {e}")
        return pd.Series({})

def extract_dominant_colors(image_location: str) -> pd.Series:
    """
    Determine the dominant colors in an image.

    Args:
    - image_location (str): The path to the image file.

    Returns:
    - pd.Series: A pandas Series containing the dominant colors and their percentages in the image.
    """
    try:
        img = Image.open(image_location)
        result = top_colors(img, 10)
        logger.info("Dominant colors extracted from image successfully")
        return result
    except Exception as e:
        logger.error(f"An unexpected error occurred while determining dominant colors: {e}")
        return pd.Series({})


def plot_dominant_colors(series: pd.Series) -> None:
    """
    Plot a pie chart showing the composition of dominant colors.

    Args:
    - series (pd.Series): A pandas Series containing the dominant colors and their percentages.
    """
    try:
        plt.figure(figsize=(8, 8))
        
        # Convert RGB values to normalized RGBA format
        colors = [(*rgb, 1) for rgb in (np.array(series.index) / 255)]

        series.plot(kind='pie', colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title('Dominant Colors Composition')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.ylabel('')
        plt.show()
    except Exception as e:
        logger.error(f"An unexpected error occurred while plotting dominant colors: {e}")


def remove_background(image_path: str, output_path: str) -> str:
    """
    Removes the background from an image and saves the result.

    :param image_path: Path to the input image file.
    :param output_path: Path to save the output image file.
    :return: Path to save the output image file.
    """
    try:
        # Check if the input image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image file '{image_path}' not found.")

      
        # Processing the image 
        input = Image.open(image_path) 
        
        # Removing the background from the given Image 
        output = remove(input) 
        
        #Saving the image in the given path 
        output.save(output_path) 

        logger.info(f"Background removed from image '{image_path}'. Result saved to '{output_path}'.")
        return output_path

    except Exception as e:
        logger.error(f"An error occurred while removing the background from image '{image_path}': {e}")
        return ""

def resize_image(image_path: str, target_width: int, target_height: int, output_path:str) -> str:
    """
    Resize an image to fit within target dimensions while maintaining aspect ratio.

    Args:
        image_path (str): The path to the image file.
        target_width (int): The desired width of the resized image.
        target_height (int): The desired height of the resized image.
        output_path (str): Path to save the output image file.

    Returns:
        PIL.Image.Image: The resized image.

    Raises:
        ValueError: If either the target width or height is non-positive.
        FileNotFoundError: If the image file does not exist.
        Exception: For any other unexpected error.
    """
    try:
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Target width and height must be positive integers.")

        image = Image.open(image_path).convert("RGBA")
        original_width, original_height = image.size

        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)

        new_height = int(original_height * ratio)
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        resized_image.save(output_path) 

        return output_path
    
    except ValueError as ve:
        logger.error(f"Error in resizing image: {ve}")
        raise ve
    except FileNotFoundError as fnfe:
        logger.error(f"Image file not found: {image_path}")
        raise fnfe
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise e


def create_combined_image(background_path: str, elements) -> Image.Image:
        """

        {
        image_path :""
        start_position_x : int
        start_position_y : int
        target_width : int
        target_height: int

        }
        Create a combined image based on background and elements' positioning and sizing.
        
        :param background_path: Path to the background image.
        :param elements: A list of dictionaries, each containing 'image_path', 'start_point', and 'dimensions'.
        """

        try:
            # Load the background image
            background = Image.open(background_path).convert("RGBA")
            
            for element in elements:
                # Load element image
                image_path = element["image_path"]            
                # Resize image according to dimensions without losing aspect ratio
                target_width, target_height = element["target_width"] ,element["target_height"]
                resized_image = resize_image(image_path, target_width, target_height)
                
                # Calculate position to center the image within its segment
                start_position_x, start_position_y = element["start_position_x"], element["start_position_y"]

                offset_x = start_position_x + (target_width - resized_image.size[0]) / 2
                offset_y = start_position_y + (target_height - resized_image.size[1]) / 2
                
                # Place the resized image on the background
                background.paste(resized_image, (int(offset_x), int(offset_y)), resized_image)
            
            logger.info("Image combined success")
            return background   

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise e
    
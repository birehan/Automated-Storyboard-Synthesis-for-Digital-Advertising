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
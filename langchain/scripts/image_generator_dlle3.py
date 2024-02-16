from openai import OpenAI
from dotenv import load_dotenv
import os
from logger import logger
from PIL import Image
import requests
from typing import  Tuple
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if not api_key:
    raise ValueError("API key is not set. Make sure it is available in your .env file.")

def generate_image_dlle3(prompt: str) -> str:
    """
    Generate an image using the OpenAI Images API based on the given prompt.

    Args:
    - prompt (str): The text prompt to generate the image.

    Returns:
    - str: The URL of the generated image.
    """
    try:
        client = OpenAI(api_key=api_key)

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            quality="hd",
            n=1,
        )

        image_url = response.data[0].url
        logger.info("Image generated successfully")

        return image_url
    except Exception as e:
        logger.error(f"Error while generating image: {e}")
        return ""



def generate_image_variation(image_src: str) -> str:
    """
    Generate variations of an input image using the OpenAI Images API.

    Args:
    - image_src (str): The path or URL of the input image.

    Returns:
    - str: The URL of the generated image variation.
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.images.create_variation(
            image=open(image_src, "rb"),
            n=2,
            size="1024x1024"
        )

        image_url = response.data[0].url
        logger.info("Image variation generated successfully")
        return image_url
    except Exception as e:
        logger.error(f"Error while generating image variation: {e}")
        return ""

def download_image_dlle3(url: str, save_path: str, image_name: str) -> Tuple[str, str]:
    """
    Downloads provided url data to given location.

    :param url: Url of the file.
    :param save_path: Folder location to save the data.
    :param image_name: Name of the image file.
    :return: Tuple of the url and save location.
    """

    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            # Get the file extension from the URL
            image_extension = ".png"
            
            # Check if the image name already exists in the save path
            existing_files = [f for f in os.listdir(save_path) if f.startswith(image_name)]
            if existing_files:
                # Append a suffix to the image name to make it unique
                image_name = f"{image_name}_{len(existing_files) + 1}"
            
            # Construct the file path with the image name and extension
            save_path = os.path.join(save_path, f"{image_name}{image_extension}")
            
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
            logger.info(f"Image saved to {save_path}")
            return (url, save_path)
        else:
            raise RuntimeError(f"Failed to download image. Status code: {response.status_code}") from None
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}") from e
    


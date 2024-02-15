import json
import requests
from typing import Tuple
from io import BytesIO
import os
from PIL import Image
import requests
from logger import logger
import io
import base64



def generate_image_automatic(prompt: str, steps: int=5, url: str = "http://localhost:7860") -> dict:
    """
    Generates images based on a prompt using a remote service.

    Args:
        prompt (str): The text prompt for generating images.
        steps (int): The number of steps in the generation process.
        url (str, optional): The URL of the remote service. Defaults to "http://localhost:7860".

    Returns:
        dict: A dictionary containing the response from the service.
    """
    payload = {
        "prompt": prompt,
        "steps": steps
    }

    try:
        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
        response.raise_for_status()  # Raise an exception for bad responses
        return response.json()

    except Exception as e:
        print(f"Error generating images: {e}")
        return {}
    

def download_image_automatic(url: str, save_path: str, image_name: str) -> Tuple[str, str]:
    """
    Downloads provided url data to given location.

    :param url: Url of the file.
    :param save_path: Folder location to save the data.
    :param image_name: Name of the image file.
    :return: Tuple of the url and save location.
    """

    try:

        image = Image.open(io.BytesIO(base64.b64decode(url)))

        if image:
            # Get the file extension from the URL
            image_extension = ".png"
            
            # Check if the image name already exists in the save path
            existing_files = [f for f in os.listdir(save_path) if f.startswith(image_name)]
            if existing_files:
                # Append a suffix to the image name to make it unique
                image_name = f"{image_name}_{len(existing_files) + 1}"
            
            # Construct the file path with the image name and extension
            save_path = os.path.join(save_path, f"{image_name}{image_extension}")
            
            image.save(save_path)
            logger.info(f"Image saved to {save_path}")
            return (url, save_path)
        else:
            raise RuntimeError("Failed to download image.") from None
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}") from e
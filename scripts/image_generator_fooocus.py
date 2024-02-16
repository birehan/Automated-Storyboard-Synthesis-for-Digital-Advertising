from typing import Literal, Optional, Tuple
import logging
import base64
from io import BytesIO
import os
from logger import logger

import replicate
from PIL import Image
import requests
from pydantic import HttpUrl
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

def generate_image_fooocus(prompt: str, image_name:str, performance_selection: Literal['Speed', 'Quality', 'Extreme Speed'] = "Extreme Speed", 
                       aspect_ratios_selection: str = "1024*1024", image_seed: int = 1234, sharpness: int = 2) -> Optional[dict]:
        """
        Generates an image based on the given prompt and settings.

        :param prompt: Textual description of the image to generate.
        :param performance_selection: Choice of performance level affecting generation speed and quality.
        :param aspect_ratio: The desired aspect ratio of the generated image.
        :param image_seed: Seed for the image generation process for reproducibility.
        :param sharpness: The sharpness level of the generated image.
        :return: The generated image or None if an error occurred.
        """
        try:
            output = replicate.run(
                "konieshadow/fooocus-api-anime:a750658f54c4f8bec1c8b0e352ce2666c22f2f919d391688ff4fc16e48b3a28f",
                input={
                    "prompt": prompt,
                    "performance_selection": performance_selection,
                    "aspect_ratios_selection": aspect_ratios_selection,
                    "image_seed": image_seed,
                    "sharpness": sharpness
                }
            )
            logging.info("Image generated successfully.")

            return download_image_fooocus(
                url = output[0],
                save_path="../generated_assets/storyboard_1/frame_1", 
                image_name=image_name)
        
        except Exception as e:
            logging.error(f"Failed to generate image: {e}")
            return None
              

def download_image_fooocus(url: str, save_path: str, image_name: str) -> str:
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
            image_extension = os.path.splitext(url)[-1]
            
            # Check if the image name already exists in the save path
            existing_files = [f for f in os.listdir(save_path) if f.startswith(image_name)]
            if existing_files:
                # Append a suffix to the image name to make it unique
                image_name = f"{image_name}_{len(existing_files) + 1}"
            
            # Construct the file path with the image name and extension
            save_path = os.path.join(save_path, f"{image_name}{image_extension}")
            
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
            logging.info(f"Image saved to {save_path}")
            return  save_path
        else:
            raise RuntimeError(f"Failed to download image. Status code: {response.status_code}") from None
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}") from e
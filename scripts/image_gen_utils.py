from openai import OpenAI
from dotenv import load_dotenv
import os
from logger import logger
import cv2

import numpy as np
import requests
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if not api_key:
    raise ValueError("API key is not set. Make sure it is available in your .env file.")

def generate_image(prompt: str) -> str:
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
    

def chat_completion(prompt: str) -> str:
    """
    Generate a chat-based completion using the OpenAI Chat API.

    Args:
    - prompt (str): The user's prompt to continue the conversation.

    Returns:
    - str: The completed content based on the user's input.

    Example:
    ```python
    user_prompt = "Tell me a joke."
    completion = chat_completion(user_prompt)
    print(completion)
    ```

    In this example, the function generates a chat-based completion based on the user's prompt.
    """
    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        logger.info("Chat completion done.")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(e)
        return ""
    


def download_image(image_url: str) -> Optional[np.ndarray]:
    """
    Download an image from a given URL, convert it to a NumPy array using OpenCV, and return the image.

    Args:
    - image_url (str): The URL of the image to be downloaded.

    Returns:
    - Optional[np.ndarray]: The NumPy array representing the downloaded image, or None if an error occurs.
    """
    try:
        # Download the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()  # Raise an HTTPError for bad responses

        # Convert to a NumPy array and then to a CV2 image
        image_data = np.frombuffer(image_response.content, np.uint8)
        background_image = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)

        logger.info("Image downloaded successfully")
        return background_image

    except requests.exceptions.RequestException as req_error:
        logger.error(f"Error during image download: {req_error}")
        return None

    except cv2.error as cv2_error:
        logger.error(f"Error decoding image with OpenCV: {cv2_error}")
        return None

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None



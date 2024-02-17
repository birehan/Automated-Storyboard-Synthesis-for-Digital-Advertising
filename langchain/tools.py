from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import List
from scripts.image_generator_fooocus import generate_image_fooocus
from scripts.image_generator_dlle3 import generate_image_dlle3
from scripts.image_analysis_utils import remove_background, resize_image, add_text_to_image, create_combined_image

class SQLQuery(BaseModel):
    query: str = Field(description="SQL query to execute")

@tool
def generate_image(prompt: str, image_name: str, save_path:str) -> str:
    """
    Generates an image based on the provided prompt using the 'generate_image_fooocus' tool.

    Args:
        prompt (str): The text prompt used for generating the image.
        image_name (str): The desired name for the generated image.
        save_path (str): The path to save the generated image. there are 3 paths "../generated_assets/storyboard_2/frame_1",  "../generated_assets/storyboard_2/frame_2" and  "../generated_assets/storyboard_2/frame_3".
        
    Returns:
        str: The local file path to the saved image.

    Example:
        generate_image("A serene landscape with mountains and a lake.", "background_image")

    """
    try:
        return generate_image_dlle3(prompt, image_name, save_path)
    except Exception as e:
        print(f"Error while generating image: {e}")
        return ""

# @tool
# def remove_image_background(image_path: str, output_path: str) -> str:
#     """
#     Removes the background from the image located at 'image_path' and saves the result to 'output_path'.

#     Args:
#         image_path (str): The file path to the input image.
#         output_path (str): The file path where the background-removed image will be saved.

#     Returns:
#         str: The file path to the saved background-removed image at 'output_path'.

#     """
#     try:
#         return remove_background(image_path, output_path)
#     except Exception as e:
#         print(f"Error while removing image background: {e}")
#         return "" 
    
    
@tool
def change_image_size(image_path: str, target_width: int, target_height: int) -> None:
    """
    Resizes the image located at 'image_path' to the specified dimensions.

    Args:
        image_path (str): Path to the input image file.
        target_width (int): Desired width of the resized image.
        target_height (int): Desired height of the resized image.

    Return:
        None
    """
    try:
        resize_image(image_path, target_width, target_height, image_path)
        None
    except Exception as e:
        print(f"Error while resizing the image: {e}")
        return ""


@tool
def insert_text_on_image(image_path: str, text: str, text_color: tuple = (255, 255, 255), font_size: int = 24, position: tuple = (10, 10), font_weight: str = "normal") -> None:
    """
    Adds text to the image located at 'image_path' with the specified attributes.

    Args:
        image_path (str): Path to the input image file.
        text (str): Text to be added to the image.
        text_color (tuple): RGB color tuple for the text (default is white).
        font_size (int): Font size (default is 24).
        position (tuple): Position where the text will be placed on the image (default is top-left corner).
        font_weight (str): Font weight ("normal" or "bold") (default is "normal").

    Returns:
        None
    """
    try:
        add_text_to_image(image_path=image_path, text=text, text_color=text_color, font_size=font_size, position=position, font_weight=font_weight)
        return
    except Exception as e:
        print(f"Error while adding text to the image: {e}")
        return ""


@tool
def combine_images_to_create_frame(background_path: str, elements: list) -> str:
    """
    Combines multiple images to create a single frame based on the background and elements' positioning and sizing.

    Args:
        background_path (str): Path to the background image.
        elements (list): A list of dictionaries, each containing information about an element:
            - 'image_path' (str): Path to the image file.
            - 'start_position_x' (int): X-coordinate of the starting position.
            - 'start_position_y' (int): Y-coordinate of the starting position.
            - 'target_width' (int): Target width of the element.
            - 'target_height' (int): Target height of the element.

    Returns:
        str: Path to the combined image.
    """
    try:
        return create_combined_image(background_path, elements)
    except Exception as e:
        print(f"Error while creating the combined image: {e}")
        return ""



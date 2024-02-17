import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)

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

        if output_path:
            resized_image.save(output_path) 

        return output_path
    
    except ValueError as ve:
        logging.error(f"Error in resizing image: {ve}")
        raise ve
    except FileNotFoundError as fnfe:
        logging.error(f"Image file not found: {image_path}")
        raise fnfe
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise e


def create_combined_image(background_path: str, elements) -> str:
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
            resized_image_path = resize_image(image_path, target_width, target_height, image_path)
            resized_image = Image.open(resized_image_path).convert("RGBA")
            
            # Calculate position to center the image within its segment
            start_position_x, start_position_y = element["start_position_x"], element["start_position_y"]

            offset_x = start_position_x + (target_width - resized_image.size[0]) / 2
            offset_y = start_position_y + (target_height - resized_image.size[1]) / 2
            
            # Place the resized image on the background
            background.paste(resized_image, (int(offset_x), int(offset_y)), resized_image)
        
        background.save(background_path)

        logging.info("Image combined success hola teosa")

        return background_path   

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise e

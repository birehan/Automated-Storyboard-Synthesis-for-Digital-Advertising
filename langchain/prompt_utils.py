

import logging
import json
from scripts.image_analysis_utils import get_image_dimensions, extract_text_on_image, extract_dominant_colors, remove_background
from scripts.object_detection import ObjectDetection

object_detector = ObjectDetection()


def load_json(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)

        return data
    except Exception as e:
        logging.error(f"Error while adding text to image: {e}")
        return None


def get_image_gen_prompt(storyboard_concept: str, storyboard_implementation: dict, storyboard_explanation: str,
                         frame_number: int, storyboard_frame: dict) -> str:
    """
    Generates prompts for generating image assets for a storyboard frame.

    Args:
    - storyboard_concept (str): The concept of the storyboard.
    - storyboard_implementation (dict): The implementation details of the storyboard frames.
    - storyboard_explanation (str): The explanation of the storyboard concept.
    - frame_number (int): The number of the storyboard frame.
    - storyboard_frame (dict): Details of the specific storyboard frame.

    Returns:
    - str: A string prompt.
    """
    prompt = f'''
        You are an advertisement assets generator. The advertisement assets are the smallest components that, 
        when combined, create a frame, and frames combined create a storyboard for the advertisement. 
        Given:
        - The storyboard concept: {storyboard_concept} 
        - The storyboard implementation: {storyboard_implementation}
        - The storyboard explanation: {storyboard_explanation}

        I want you to generate asset images for storyboard frame {frame_number}.
        Given storyboard_frame_{frame_number}: {storyboard_frame}, the storyboard frame_{frame_number} contains 
        the asset category paired with the prompt that generates the image. Given the content, update the prompts 
        to become more effective and direct, specify the specific color to be used, components, texture, and make 
        sure the prompt does not generate an image including any text; no text should be in the image generation.
        Return a JSON object of pairs of asset categories with image file paths.
        
        The background image size should be specified.
        '''
    logging.info("Generated prompt successfully.")
    return prompt
   


def get_frame_images_detail(asset_response: dict) -> list:
    """
    Retrieves details of frame images including dimensions, dominant colors, detected objects, and extracted text.

    Args:
    - asset_response (dict): Dictionary containing pairs of asset categories with image file paths.

    Returns:
    - list: List of dictionaries containing image details.
    """
    details = []

    try:
        for asset_name, image_path in asset_response.items():
            width, height = get_image_dimensions(image_path)
            dominant_colors = list(dict(extract_dominant_colors(image_path)).items())[:3]

            if asset_name != "Background":
                remove_background(image_path, image_path)

            extracted_text = extract_text_on_image(image_path)
            detected_objects = object_detector.detect_objects_and_info(image_path)

            details.append({
                "asset_category": asset_name,
                "image_path": image_path,
                "image_width": width,
                "image_height": height,
                "extracted_text_on_image": extracted_text,
                "detected_objects_in_image": detected_objects,
                "dominant_colors_on_image": dominant_colors
            })
        
        logging.info("Retrieved frame images details successfully.")
        return details
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return []

def get_frame_gen_prompt(storyboard_concept: str, storyboard_implementation: dict, 
                                 storyboard_explanation: str, frame_number: int, prompt: str, 
                                 asset_response: dict) -> str:
    """
    Generates a prompt for creating an advertisement frame by combining assets.

    Args:
    - storyboard_concept (str): The concept of the storyboard.
    - storyboard_implementation (dict): The implementation details of the storyboard frames.
    - storyboard_explanation (str): The explanation of the storyboard concept.
    - frame_number (int): The number of the storyboard frame.
    - prompt (str): The prompt for generating assets.
    - asset_response (dict): Dictionary containing pairs of asset categories with image file paths.

    Returns:
    - str: A prompt guiding the generation of the advertisement frame.
    """
    prompt = f'''
        You are an advertisement frame generator. The advertisement assets are the smallest components that, 
        when combined, create a frame, and frames combined create a storyboard for the advertisement. 
        Given:
        - The storyboard concept: {storyboard_concept} 
        - The storyboard implementation: {storyboard_implementation}
        - The storyboard explanation: {storyboard_explanation}
        - The storyboard asset generation prompt: {prompt}

        The assets have already been generated, which will help us to create the frame. The generated image 
        response I have is here: {asset_response}

        First and foremost, resize the background image to 320 * 560.

        I want you to combine the assets for the frame {frame_number}. You are going to create {asset_response}.
        Combine all the images to create one image. You have to resize images other than the background into smaller 
        elements and place them in a good position to make it more attractive and appealing. Add text for images 
        needing text to make the advertisement frame descriptive and precise, and add a short, descriptive text on 
        the background. Add Text Elements to the background; make the color compatible with the background and 
        proportional to the background size. Do not call the tool 'generate_image'; the end goal is to combine the 
        images.
        - Invoke the function 'resize_image' for all images other than the background image and give them a 
        suitable size.
        - Invoke 'add_text_on_image' to add relevant text on images.
        - Invoke 'combine_images_to_create_frame' to get the combined image frame.

        Arrange the different elements in a good position and ensure they do not overlap.

        Return only the combined image URL.
        '''
    logging.info("Generated frame generation prompt successfully.")
    return prompt
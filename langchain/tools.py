from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import List
from scripts.image_generator_fooocus import generate_image_fooocus
from scripts.image_analysis_utils import remove_background, resize_image

class SQLQuery(BaseModel):
    query: str = Field(description="SQL query to execute")

@tool
def generate_image(prompt: str, image_name: str) -> str:
    """
    Generates an image based on the provided prompt using the 'generate_image_fooocus' tool.

    Args:
        prompt (str): The text prompt used for generating the image.
        image_name (str): The desired name for the generated image.

    Returns:
        str: The local file path to the saved image.

    Example:
        generate_image("A serene landscape with mountains and a lake.", "background_image")

    """
    try:
        return generate_image_fooocus(prompt, image_name)
    except Exception as e:
        print(f"Error while generating image: {e}")
        return ""

@tool
def remove_image_background(image_path: str, output_path: str) -> str:
    """
    Removes the background from the image located at 'image_path' and saves the result to 'output_path'.

    Args:
        image_path (str): The file path to the input image.
        output_path (str): The file path where the background-removed image will be saved.

    Returns:
        str: The file path to the saved background-removed image at 'output_path'.

    """
    try:
        return remove_background(image_path, output_path)
    except Exception as e:
        print(f"Error while removing image background: {e}")
        return ""
    
    
@tool
def change_image_size(image_path: str, output_path: str, target_width:str, target_height:str) -> str:
    """
    """
    try:
        return resize_image(image_path, target_width, target_height, output_path)
    except Exception as e:
        print(f"Error while removing image background: {e}")
        return ""
    


        


# remove_background
# @tool
# def generate_evaluation_data(query: str) -> List:
#     """Returns geneated evaluation data"""
#     return '''
# Your task is to formulate exactly the number of questions taken from user {query} or set it to 5 questions from given context and provide the answer to each one.

# End each question with a '?' character and then in a newline write the answer to that question using only 
# the context provided.
# The output MUST BE in a json format. 

# example:
# [
# {
#     "user": "What is the name of the company?",
#     "assistant": "Google"
# },
# {
#     "user": "What is the name of the CEO?",
#     "assistant": "Sundar Pichai"
# }
# ]

# Each question must start with "user:".
# Each answer must start with "assistant:".


# The question must satisfy the rules given below:
# 1.The question should make sense to humans even when read without the given context.
# 2.The question should be fully answered from the given context.
# 3.The question should be framed from a part of context that contains important information. It can also be from tables,code,etc.
# 4.The answer to the question should not contain any links.
# 5.The question should be of moderate difficulty.
# 6.The question must be reasonable and must be understood and responded by humans.
# 7.Do no use phrases like 'provided context',etc in the question
# 8.Avoid framing question using word "and" that can be decomposed into more than one question.
# 9.The question should not contain more than 10 words, make of use of abbreviation wherever possible.
    
# context: taken the context from the challenge document provided in the retriver according to user query {query}
# '''

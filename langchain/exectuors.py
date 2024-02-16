from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import AgentType, initialize_agent
from langchain.schema import SystemMessage
import logging
from tools import dummy_tool

logging.basicConfig(level=logging.INFO)
# from tools import generate_prompts_with_evaluation, get_prompt_ranking_monte_carol_and_elo_rating, generate_evaluation_data

with open("system_message.txt", "r") as file:
    system_message = file.read()

# def get_agent_executor(model_name='gpt-4-1106-preview',temperature=0):

#     try:
#         llm = ChatOpenAI(model_name=model_name, temperature=temperature)

#         # Define prompt template
#         prompt = ChatPromptTemplate.from_template(system_message)

#         # Setup Langchain pipeline
#         lang_chain = (
#             {"question": RunnablePassthrough()} 
#             | prompt 
#             | llm
#             | StrOutputParser() 
#         )

#         logger.info("langchain  created successfully.")

#         return lang_chain
    
#     except Exception as e:
#         logger.error(f"An unexpected error occurred while creating langchain exectuor: {e}")
#         return None 


def get_agent_executor(model_name='gpt-4-1106-preview', temperature=0):
    try:
        agent_kwargs = {
        "system_message": SystemMessage(content=system_message),
        }

        analyst_agent_openai = initialize_agent(
            llm=ChatOpenAI(temperature=temperature, model = model_name),
            agent=AgentType.OPENAI_FUNCTIONS,
            tools=[dummy_tool],
            agent_kwargs=agent_kwargs,
            verbose=True,
            max_iterations=20,
            early_stopping_method='generate'
        )

        logging.info("langchain  created successfully.")
        return analyst_agent_openai

    except Exception as e:
        logging.error(f"An unexpected error occurred while creating langchain exectuor: {e}")
        return None
    
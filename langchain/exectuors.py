from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.schema import SystemMessage
import logging
from tools import generate_image

logging.basicConfig(level=logging.INFO)

with open("system_message.txt", "r") as file:
    system_message = file.read()


def get_agent_executor(model_name='gpt-4-1106-preview', temperature=0):
    try:
        agent_kwargs = {
        "system_message": SystemMessage(content=system_message),
        }

        analyst_agent_openai = initialize_agent(
            llm=ChatOpenAI(temperature=temperature, model = model_name),
            agent=AgentType.OPENAI_FUNCTIONS,
            tools=[generate_image],
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
    
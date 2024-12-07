import os
from openai import OpenAI
from LLM_API import get_chatgpt_answer



message = get_chatgpt_answer("bla bla ")
print(message)
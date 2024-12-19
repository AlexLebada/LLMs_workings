import os
from openai import OpenAI
from LLM_API import get_chatgpt_answer



message = get_chatgpt_answer("tell me where a cat can live ", 5)

#print(message)
for i, answer in enumerate(message, 1):
    print(f"Answer {i}:\n{answer}\n")
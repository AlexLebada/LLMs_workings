from openai import OpenAI
import pandas as pd
import os


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


#Calculate word embeddings => vectors
def get_chatgpt_embedding(text_to_embed):
    df = pd.read_csv('../LLM_dashboard.csv')
    csv_model = "text-embedding-ada-002"
    df.at[0, csv_model] += 1
    df.to_csv('../LLM_dashboard.csv', index=False)
    response = client.embeddings.create(
        model= csv_model,
        input=[text_to_embed]
    )
    return response.data[0].embedding  # Change this



def get_chatgpt_answer(input_text: str, input_no_answers: int):
    df = pd.read_csv('../LLM_dashboard.csv')
    req_answers = input_no_answers
    csv_tokens_used = 50
    csv_model = 'gpt-4o'
    df.at[2, csv_model] += input_no_answers
    df.at[1, csv_model] += (csv_tokens_used * input_no_answers)
    df.to_csv('../LLM_dashboard.csv', index=False)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_text,
            }
        ],
        model=csv_model,
        max_tokens=csv_tokens_used,
        temperature=0.7,
        top_p=0.5,
        n=input_no_answers,
    )

    answers = [choice.message.content for choice in response.choices]
    return answers



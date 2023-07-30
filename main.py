import os

import dotenv
import openai
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd

description ="""
There are 17 elvin families. Each family lives in its own forest.
elves table: elf_id, parent_id, name, age, savings, alive, family_id
forests table: family_id, forest_name, forest_square
family_id links elves to forests. One-to-many relationship from forests to elves:
each forest has multiple elves (a family), each elf has one forest.
"""

def main():

    df_elves = pd.read_csv("data/elf_families_small.csv")
    df_forests = pd.read_csv("data/elf_forests.csv")
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), [df_elves, df_forests], verbose=True)

    print(description)
    while True:
        prompt = input("Ask chatbot about elves: ")
        if prompt.lower() == "quit":
            exit(0)
        else:
            agent.run(prompt)

def setup_openai():
    dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        openai.api_base = api_base


if __name__ == "__main__":
    setup_openai()
    main()
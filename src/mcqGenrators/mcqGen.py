import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqGenrators.utils import read_file
# ,get_table_data
from src.mcqGenrators.logger import logging


from langchain.llms import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2


load_dotenv()

key=os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(openai_api_key=key,model_name="gpt-3.5-turbo",temperature=0.7)

TEMPLATE = """
You are an expert educator and assessment designer.

Your task is to generate {number} high-quality multiple-choice questions (MCQs) for the subject: "{subject}". The questions should be based on the following content or context:

"{text}"

Guidelines:
- The tone of the questions and answers should be {tone}.
- Each question must have 1 correct answer and 3 plausible but incorrect distractors.
- Questions should test meaningful understanding of the topic and not be trivially easy.
- Avoid repeating options across questions.

Respond strictly in the following JSON format:

{response_json}
"""
quiz_generation_mcqs=PromptTemplate(
  input_variables=["text","number","subject","tone","response_json"],
  template=TEMPLATE
)
quiz_chain=LLMChain(llm=llm,prompt=quiz_generation_mcqs,output_key="quiz",verbose=True)

TEMPLATE2 = """You are an expert educator and assessment designer.

Your task is to critically evaluate the quality of the following multiple-choice questions (MCQs) for the subject: "{subject}".

Evaluate the questions using these criteria:
1. Does each question test a meaningful concept or higher-order thinking?
2. Is there exactly 1 correct answer and 3 plausible but incorrect distractors?
3. Are the options clear, non-repetitive, and free of ambiguity?
4. Does the tone match the intended tone: {tone}?
5. Are the questions aligned with the given content/context?

Here is the content/context:
"{text}"

Here are the generated MCQs to evaluate:
{quiz}

Instructions:
- If the MCQs meet all quality criteria, respond with a JSON indicating they are valid and include no changes.
- If one or more MCQs are poor or substandard, revise **only those questions**. Keep strong questions unchanged.
- Ensure all revisions follow the original formatting and quality guidelines.

"""
quiz_evaluation_prompt=PromptTemplate(input_variables=["subject","tone","text","quiz"],template=TEMPLATE2)

review_chain=LLMChain(llm=llm,prompt=quiz_evaluation_prompt,output_key="review",verbose=True)

generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone","response_json"],  # added
    output_variables=["quiz", "review"],
    verbose=True
)

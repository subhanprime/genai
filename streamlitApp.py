import os 
import json
import pandas as pd
import traceback
import streamlit as st

from dotenv import load_dotenv
from src.mcqGenrators.utils import read_file
from langchain.callbacks import get_openai_callback
from src.mcqGenrators.mcqGen import generate_evaluate_chain
from src.mcqGenrators.logger import logging

response_file_path = os.path.join('mcqgen', 'response.json')
with open(response_file_path, 'r') as file:
  RESPONSE_JSON=json.load(file)
  
  # creating a title for stream
  st.title("MCQ APP GENERATOR")
  
  # create a form using st.form
  with st.form("user_input"):
    # file upload
    uploaded_file=st.file_uploader("upload a txt or pdf file")
    
    mcq_count=st.number_input("No of mcq",min_value=3,max_value=30)
    
    subject=st.text_input("entre sub", max_chars=30)
    
    tone=st.text_input("complexity of question",max_chars=20 ,placeholder="simple")
    
    button=st.form_submit_button("create mcq")
    
  if button and uploaded_file is not None and mcq_count and subject and tone:
    with st.spinner("loading..."):
      try:
        text=read_file(uploaded_file)
        with get_openai_callback() as cb:
          response=generate_evaluate_chain({
            "text":text,
            "number":mcq_count,
            "subject":subject,
            "tone":tone,
            "response_json":json.dumps(RESPONSE_JSON)
          })
          
      except Exception as e:
        traceback.print_exception(type(e),e, e.__traceback__)
        st.error("Error")
        
      else:
        print(f"number of token: {cb.total_tokens}")
        if (isinstance, dict):
          quiz=response.get("quiz")
          if quiz is not None:
            st.text_area(label="review",value=response["quiz"])

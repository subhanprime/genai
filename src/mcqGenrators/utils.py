import os
import PyPDF2
import json
import traceback

def read_file(file):
  if file.name.endswith(".pdf"):
    try:
      pdf_reader=PyPDF2.PdfFileReader(file)
      text=""
      for page in pdf_reader.pages:
        text+=page.extract_text()
        return text
    except Exception as e:
      raise Exception("error reading the file pdf reader")
    
  elif file.name.endswith(".txt"):
    return file.read().decode("utf-8")
  
  else:
    raise Exception("unsupported file format only read txt and pdf")
  
  # def get_table_data(quiz_str):
  #   try:
  #     quiz_dict=json.load(quiz_str)
  #     quiz_table_data=[]
      
  #     for key,value, in quiz_dict.items():
  #       mcq=value[mcq]
        
    
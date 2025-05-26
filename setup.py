from setuptools import setup, find_packages


setup(
    name="mcqgen",
    version="0.1",
    packages=find_packages(),
    install_requires=["openai","langchain","streamlit","python_dotenv","PyPDF2"],
)
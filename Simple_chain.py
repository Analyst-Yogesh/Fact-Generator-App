import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import login

# ------------------------
# API Key Management
# ------------------------
if os.path.exists(".env"):  # Local Development
    load_dotenv()
    hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
else:  # Streamlit Cloud
    hf_api_key = st.secrets["huggingface"]["api_key"]

# Login to Hugging Face Hub
login(hf_api_key)

# ------------------------
# LangChain Setup
# ------------------------
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=hf_api_key
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
chain = prompt | model | parser

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Fact Generator", page_icon="✨")

st.title("✨ Fact Generator App")
st.write("Enter a topic and get 5 interesting facts!")

topic = st.text_input("Enter a topic:", "")

if st.button("Generate Facts"):
    if topic.strip():
        with st.spinner("Generating facts..."):
            result = chain.invoke({"topic": topic})
        st.subheader("Generated Facts:")
        st.write(result)
    else:
        st.warning("⚠️ Please enter a topic first!")

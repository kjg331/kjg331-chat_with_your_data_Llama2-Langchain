import streamlit as st

import subprocess

import json
import requests
import time

from huggingface_hub import hf_hub_download

from langchain import hub
from langchain import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WebBaseLoader
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import StreamlitChatMessageHistory

from googletrans import Translator
import bs4

DB_FAISS_PATH = 'vectorstore/db_faiss'

def main():
    st.set_page_config(page_title="Naver News Chatbot - An LLM powered Chatbot for QA with Naver News")

    # Sidebar contents
    with st.sidebar:
        st.title('💬 Naver News Chatbot')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) LLM model
        
        'Made by [Kim JinGu](https://github.com/kjg331)'
            
            
            
        💡 Note: Should submit the Hugging Face Token key first!
        ''')
            
            
        token_hugging_face = st.text_input("Enter your Hugging Face token", type="password")
        process = st.button("Process")
 
            
        if process:
            if token_hugging_face.startswith("hf_"):
                command = f"huggingface-cli login --token {token_hugging_face}"
                try:
                    subprocess.run(command, shell=True, check=True)
                    st.write("Success: Command executed successfully!")
                except subprocess.CalledProcessError as e:
                    st.write(f"Error: {e}")
            else:
                st.write("Failure: Please make sure your Hugging Face token is valid")
                    
            #st.subheader('Parameters')
            #temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
            #max_length = st.sidebar.slider('max_length', min_value=32, max_value=4000, value=120, step=8)
            
        # Inputs
        if "processComplete" not in st.session_state:
            st.session_state.processComplete = None
        
        st.subheader("Enter the URL of the article you want to chat")

        if "url_data" not in st.session_state:
            st.session_state.url_data = None

        url = st.text_input("URL:")
        copy_button = st.button("Copy the address of articles")

        # URL 추가 및 중복 방지
        if copy_button and url:
            st.session_state.url_data = url
            st.session_state.url_data = list(set(st.session_state.url_data))

        # 현재 URL 표시
        if st.session_state.url_data:
            st.caption("Current URLs:")
            st.write(f"{url}")
            delete_button = st.button("Delete")
            if delete_button:
                del st.session_state.url_data
                st.experimental_rerun()

        else:
            st.write("No URLs added yet.")
    
        # Finalize and return the list of URLs
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        
        if st.button("Submit"):
            if not token_hugging_face:
                st.write("Please add your Hugging Face Token to continue")
            
            if not st.session_state.url_data:
                st.write("You didn't submit URL yet")
                
            st.caption("URL submission completed.")
            
            # Make the chain bot with the given URLs        
            st.session_state.conversation = load_db(url)
            st.session_state.processComplete = True

        def clear_chat_history():
            st.session_state.messages = [{"role": "assistant", "content": "뉴스에 대해 궁금한게 있으신가요?"}]

        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)       
    

                    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "뉴스에 대해 궁금한게 있으신가요?"}]
    
    if st.session_state.get("chat_history") is None:
        st.session_state.chat_history = []
              
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")           

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
                response = result['answer'] 
                ko_response = ko_trans(response)
                st.markdown(ko_response)
                
                st.session_state.chat_history.extend([(query, response)])
               
                source_documents = result['source_documents']
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                        
        # Add the chat history
        st.session_state.messages.append({"role": "assistant", "content": ko_response})   

def load_db(file):
    # load documents
    loader = WebBaseLoader(
    file,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("media_end_head_title", "media_end_head_info_datestamp_time _ARTICLE_DATE_TIME", "newsct_article _article_body")
        )
    ),
)
    documents = loader.load()

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # define embedding
    embeddings =  HuggingFaceEmbeddings(
                                        model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
    )

    # create vector database from data
    db = None
    db = FAISS.from_documents(docs, embeddings)

    # define retriever
    retriever = db.as_retriever(search_type="similarity", vervose = True, search_kwargs={"k": 3})

    # create a chatbot chain. Memory is managed externally.

    model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_basename = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"

    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

    qa = None
    qa = ConversationalRetrievalChain.from_llm(
        llm=LlamaCpp(model_path=model_path,
                     max_tokens= 800,
                     #n_gpu_layers = 40,
                     #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                     n_ctx=4096, # Context window
                     verbose = True,
                     temperature=0.1,
                     repeat_penalty=1.2),
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa
    
def ko_trans(sentence):
    translator = Translator()
    return  translator.translate(sentence, dest='ko').text
    
    
if __name__ == '__main__':
    main()
    

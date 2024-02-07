This is the RAG[https://python.langchain.com/docs/use_cases/question_answering/] system by using Llama.cpp(Mistral Mainly) to chat with the your own data.

Streamlit file is the specified APP to chat with the naver news.
What makes this APP sepcial is that it gives the asnwer as Koraen with the small model of Mistral(7B) by utilizing the googletranse library.
It makes the APP faster response with the higher quality of the original generated English answer(Most of the small LLM return the English answer because of the lack of pre-traiend language tokens).
In addition this is free APP if you have a Hugging Face Toekn of Llama2. 

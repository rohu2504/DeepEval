import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool 
from dotenv import load_dotenv

load_dotenv()


embed_model="sentence-transformers/all-MiniLM-L6-v2"
api_key=os.getenv("groq_api_key")
llm=ChatGroq(model_name="openai/gpt-oss-120b",api_key=api_key)
load_doc=PyPDFLoader("deepeval/data/diabetes.pdf")
doc=load_doc.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=30)
split=splitter.split_documents(doc)

embed=HuggingFaceEmbeddings(model_name=embed_model)

vec_store=FAISS.from_documents(split,embed)

retriever=vec_store.as_retriever(search_kwargs={"k":4})

# resp=retriever.invoke("What is Diabetes")


def format_(doc):

    get_=[i.page_content for i in doc]
    club_="\n".join(get_)
    return club_


context_send=""

@tool
def retriever_fun(query:str):
    '''
    retrieves Answer from the Knowledge base for answering the query asked

    Args:

    query:Question for which answer need to be retrieved from the knowledge base.

    '''
    
    

    global context_send
    retrieve=retriever.invoke(query)
    formatt=format_(retrieve)
    context_send+=formatt
    
    return formatt


prompt=f"""

You are an intelligent and helpful question and answering agent your task is to answer the query asked.

<Steps to be followed>

1.Understand the question that has been provided.

2.After that inorder to answer the query that has been asked strictly use the retriever_fun tool to answer the query.

3.To answer the query strictly use the tool and response returned by the tool to answer the query.

4.Don't provide any preamble or thinking in the response.

5.Make sure the response to be clear and the response should be provided using the response returned by the tool only.

</Steps to be followed>

Note: Strictly follow all the instructions carefully.

"""

agent=create_agent(model=llm,system_prompt=prompt,tools=[retriever_fun])



dic={}
tool_call=[]
def run_agent(query):

    respo=agent.invoke({"messages":query})
    # print(respo["messages"])
    # print("--------------------------------------------------------------------------")
    tool_=respo["messages"][1].additional_kwargs["tool_calls"][0]["function"]["name"]
    fi_respo=respo["messages"][-1].content
    dic["context"]=context_send
    dic["response"]=fi_respo
    dic["tool_call"]=tool_
    return dic


if __name__=="__main__":

    query=input("Please specify your query?")
    print(run_agent(query))






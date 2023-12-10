from langchain.llms import OpenAI #Import openai
from dotenv import load_dotenv #Import dotenv
from langchain.document_loaders.csv_loader import CSVLoader #Load csv data
from langchain.embeddings import OpenAIEmbeddings #Import openai embeddings
from langchain.text_splitter import CharacterTextSplitter  #Split characters
from langchain.vectorstores import FAISS #FAISS - VectorDB
from langchain.chains import RetrievalQA #Import RQA
from langchain.prompts import PromptTemplate #Import Prompt Template
import os #import os

load_dotenv() #initialize dotenv

# Test the imports
print("Imports are working!")

class RegModel:
    def __init__(self):

        # Create a folder for data if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")
    
        # Load documents from the data folder
        self.data = CSVLoader(file_path= './data/codebasics_faqs.csv', source_column="prompt").load()

        # Initialize LLM
        self.llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.9, max_tokens=500)

        # Initialize Embeddings
        self.embeddings = OpenAIEmbeddings()

        # Split data into chunks
        self.docs = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0).split_documents(self.data)

        # Create VectorDB
        self.db = FAISS.from_documents(self.docs, self.embeddings)

        # Create a retriever
        retriever = self.db.as_retriever(score_threshold = 0.7)

    def getChain(self, context, question): 
        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""


        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}

        chain = RetrievalQA.from_chain_type(llm=self.llm,
                                            chain_type="staff", #map_reduce
                                            retriever=self.retriever,
                                            input_key="query",
                                            return_source_documents=True,
                                            chain_type_kwargs=chain_type_kwargs) 
        
        return chain
    
if __name__ == "__main__":
    model = RegModel()
    chain = model.getChain()
    chain("do you have blockchain course")






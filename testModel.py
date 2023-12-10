from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# Set OpenAI API key directly in the script
os.environ["OPENAI_API_KEY"] = "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"

embeddings = OpenAIEmbeddings()

vectordb_file_path = "faiss_index"


class RegModel:
    def __init__(self):
        # Create a folder for data if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")

        # Load documents from the data folder
        self.data = CSVLoader(file_path='./data/codebasics_faqs.csv', source_column="prompt").load()

        # Initialize LLM
        self.llm = OpenAI(model_kwargs={"open_api_key": os.environ["OPENAI_API_KEY"]}, temperature=0.9, max_tokens=500)

        # Initialize Embeddings
        self.embeddings = OpenAIEmbeddings()

        # Split data into chunks
        self.docs = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0).split_documents(self.data)

        # Create VectorDB
        self.db = FAISS.from_documents(self.docs, self.embeddings)

        # Create a retriever
        self.retriever = self.db.as_retriever(score_threshold=0.7)

    import re

    class RegModel:
        # ... (rest of your class)

        def getChain(self):
            prompt_template = """Given the following context and a question, generate an answer based on this context only.
            In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
            If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

            CONTEXT: {context}

            QUESTION: {question}"""

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            chain_type_kwargs = {"prompt": PROMPT}

            # Create a simple regex parser
            output_parser = re.compile(r'Answer: (.*)')  # Adjust the regex pattern based on your expected output

            chain = RetrievalQA.from_chain_type(llm=self.llm,
                                                chain_type="map_rerank",
                                                retriever=self.retriever,
                                                input_key="query",
                                                return_source_documents=True,
                                                chain_type_kwargs=chain_type_kwargs,
                                                output_parser=output_parser)  # Provide the output_parser

            return chain


if __name__ == "__main__":
    model = RegModel()
    chain = model.getChain()
    chain("do you have blockchain course")

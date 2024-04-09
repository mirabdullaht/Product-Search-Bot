# from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase #connecting to SQL database
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS #vector embeddngs
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from sqlalchemy import create_engine

from few_shots import few_shots

import os

load_dotenv()

#Function to retrieve the define the query and retrieve the result from the DB
def get_few_shot_db_chain():
    #loading g the llm
    llm=Ollama(model = 'mistral')

    #Connecting to the MySQL Database on Local
    #db_password: /Do Not Use Special Characters/
    db_user = "root"
    db_password = "admin"
    db_host = "localhost:3306"
    db_name = "atliq_tshirts"

    db_engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", pool_pre_ping=True)
    db = SQLDatabase(db_engine, sample_rows_in_table_info=3)

    #Embeddings
    embeddings=OllamaEmbeddings(model='mistral')
    # creating a blob of all the sentences
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    #generating a vector store: 
    vector_store=FAISS.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    #example selector to check sematic similarity
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore = vector_store,
        k=2, #number of examples
    )

    #Prompt Prefix
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".

    Use the following format:

    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here

    No pre-amble.
    """

    #Example prompt format
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    # Entire Prompt
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )

    #Chain_: 
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)

    return chain


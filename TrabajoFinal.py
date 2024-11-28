#-- IMPORTACION DE LIBRERIAS --#
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from dotenv import load_dotenv
import os

#-- CONFIGURO MI VARIABLE DE ENTORNO PARA LLAMAR A LAS API --#
load_dotenv()

#-- CARGA DE LA CARPETA QUE CONTIENE LOS PDFs QUE SON LA BASE DE CONOCIMIENTO --#
loader = DirectoryLoader(
    "./pdf_tif",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)
documents_pages = loader.load()

docs =[]
docs_lazy = loader.lazy_load()
for doc in docs_lazy:
    docs.append(doc)

#-- PARTICION DE LOS PDF EN FRAGMENTOS DE 400 CARACTERES --#

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 400,
    chunk_overlap=40,
    separators=["\n", "\n", "."]
)

documents = splitter.split_documents(docs)

#-- CONFIGURACION DEL MODELO DE EMBEDDING Y LLM A UTILIZAR, OPENAI EN ESTE CASO --#

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

#-- CREACION DE LA BASE DE DATOS VECTORIALES --#

vectorstore = FAISS.from_documents(documents, embeddings) #Creamos la BD vectorial a partir de los PDFs fragmentados

retriever = vectorstore.as_retriever()

#-- PLANTILLA DEL PROMPT QUE UTILIZAREMOS --#

template = """Actuar como un experto asistente en materia de normativa docente de la provincia de 
misiones, república de argentina. Tus destinatarios son docentes y personal administrativo de escuelas
de nivel inicial, primario y secundario. Debes responder las preguntas que te hagan, basándote 
exclusivamente en la normativa proporcionada. Si no sabes la respuesta, simplemente di que no lo sabes.
El estilo que debes usar para dar la respuesta es formal. 
{context}

Pregunta:{question}
"""
prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

#-- ELABORACION DE LA INTERFAZ DE USUARIO --#

st.title("Normativa docente - Provincia de Misiones")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Escribe tu pregunta: "):
    st.session_state.messages.append({"role": "user", "content": query})

with st.chat_message("user"):
    st.markdown(query)

with st.chat_message("assistant"):
    with st.spinner("Procesando..."):
        message_placeholder = st.empty()
        full_response = ""

        message = chain.invoke(query)
        full_response += message

        message_placeholder.markdown(full_response + " |")

        message_placeholder.markdown(full_response)
        
st.session_state.messages.append({"role": "assistant", "content": full_response})
import io
import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM
import google.genai as genai
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from typing import Optional

load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")


@st.cache_resource
def load_database():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


def load_uploaded_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    reader = PdfReader(io.BytesIO(file_bytes))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_content = page.extract_text() or ""
        if page_content.strip():
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_number,
                    },
                )
            )

    if not documents:
        raise ValueError(
            f"{uploaded_file.name} contains no extractable text. Scanned PDFs "
            "require OCR before they can be added."
        )

    return documents


def add_uploaded_pdfs(uploaded_files):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    documents = []
    for uploaded_file in uploaded_files:
        documents.extend(load_uploaded_pdf(uploaded_file))

    chunks = text_splitter.split_documents(documents)
    vectorstore = load_database()
    vectorstore.add_documents(chunks)
    vectorstore.save_local(DB_FAISS_PATH)
    return len(chunks)


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )


# configure the Google Generative API client from env
GENAI_CLIENT = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

class GeminiLLM(LLM):
    """LangChain-compatible Gemini wrapper (Pydantic style)."""
    model_name: str = "gemini-3.5-flash-lite"
    temperature: float = 1.0
    max_output_tokens: Optional[int] = None

    def _call(self, prompt: str, stop=None) -> str:
        response = GENAI_CLIENT.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
        return response.text

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name, "temperature": self.temperature}

    @property
    def _llm_type(self):
        return "gemini"

def setup_gemini_llm(model_name="gemini-3.5-flash-lite", temperature=1.0):
    # ensure the client is configured (load_dotenv already ran earlier)
    return GeminiLLM(model_name=model_name, temperature=temperature)


def main():
    st.title("Ask Chatbot!")

    with st.sidebar:
        if "upload_widget_version" not in st.session_state:
            st.session_state.upload_widget_version = 0

        st.subheader("Add documents")
        st.caption(
            "Uploaded PDFs are added to the shared knowledge base and can be "
            "used by all users of this app."
        )
        if "upload_success_message" in st.session_state:
            st.success(st.session_state.pop("upload_success_message"))

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"knowledge_base_upload_{st.session_state.upload_widget_version}",
        )
        if st.button("Add to knowledge base", disabled=not uploaded_files):
            with st.status("Adding documents to the knowledge base...", expanded=True) as status:
                st.write("Extracting PDF text, creating embeddings, and saving the shared index.")
                try:
                    chunk_count = add_uploaded_pdfs(uploaded_files)
                except Exception as exc:
                    status.update(label="Unable to add uploaded documents", state="error")
                    st.error(f"Unable to add uploaded documents: {exc}")
                else:
                    status.update(label="Documents added to the knowledge base", state="complete")
                    st.session_state.upload_success_message = (
                        f"Added {len(uploaded_files)} PDF(s) as {chunk_count} searchable text chunks."
                    )
                    st.session_state.upload_widget_version += 1
                    st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_prompt = st.chat_input("Pass your prompt here")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        custom_prompt_template = """
            Use the pieces of information provided in the context to answer the user's question.
            If you don't know the answer, say that you don't know and do not make up an answer.
            Do not provide anything outside the given context.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk.
        """

        try:
            vectorstore = load_database()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=setup_gemini_llm(model_name="gemini-3.5-flash-lite", temperature=1.0),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)},
            )

            response = qa_chain.invoke({"query": user_prompt})
            result = response["result"]
            source_documents = response.get("source_documents", [])
        #   result_to_show = result + "\nSource Docs:\n" + str(source_documents)
            result_to_show = result
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as exc:
            st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()

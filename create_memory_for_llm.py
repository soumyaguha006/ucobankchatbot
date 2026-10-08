import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'python-dotenv'. Activate the project virtual environment or install dependencies with: pip install -r requirements.txt"
    ) from exc

try:
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing one or more LangChain dependencies. Activate the project virtual environment or install dependencies with: pip install -r requirements.txt"
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"

load_dotenv(BASE_DIR / ".env")


def load_pdf_files(data_dir: Path) -> list:
    """Load all PDF files under the given directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory for PDF data: {data_dir}")

    loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDF files were found in: {data_dir}")
    return documents


def create_chunks(extracted_data, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split extracted PDF content into chunked documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(extracted_data)


def create_embeddings():
    """Create the Hugging Face embedding model used for FAISS indexing."""
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_kwargs = {"device": "cpu"}
    if hf_token:
        model_kwargs["token"] = hf_token

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
    )


def create_vector_store(chunks, embedding_model, output_dir: Path):
    """Create and persist a FAISS vector store from the chunked documents."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local(str(output_dir))
    return vector_store


def build_memory(data_dir: Path = DATA_PATH, output_dir: Path = DB_FAISS_PATH, chunk_size: int = 500, chunk_overlap: int = 50):
    """Build the FAISS index from all PDFs found in the input directory."""
    documents = load_pdf_files(data_dir)
    print(f"Length of PDF pages: {len(documents)}")

    chunks = create_chunks(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Length of text chunks: {len(chunks)}")

    embedding_model = create_embeddings()
    vector_store_final = create_vector_store(chunks, embedding_model, output_dir)
    print(f"Vector store saved at: {output_dir}")
    return vector_store_final


def parse_args():
    parser = argparse.ArgumentParser(description="Build a FAISS memory index from PDF documents.")
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH, help="Directory containing the source PDFs.")
    parser.add_argument("--output-dir", type=Path, default=DB_FAISS_PATH, help="Directory where the FAISS index will be saved.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Max chunk size for text splitting.")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between consecutive text chunks.")
    return parser.parse_args()


def main():
    args = parse_args()
    build_memory(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()

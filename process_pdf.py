from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = r"C:\\Users\\veliv\\Downloads\\Remote_Vehicle_Position_Classification.pdf"

loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"Number of pages: {len(pages)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
    )

docs = splitter.split_documents(pages)

print(f"Number of chunks: {len(docs)}")
print(f"First chunk: {docs[0].page_content}")
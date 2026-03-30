from langchain_ollama import OllamaLLM

llm = OllamaLLM(model = "llama3:instruct")
print(llm.invoke("What is a research paper?"))
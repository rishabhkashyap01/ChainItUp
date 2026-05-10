"""
RAG (Retrieval-Augmented Generation) - Document Q&A

This module demonstrates RAG implementation in LangChain.
Learn how to:
1. Load documents from various sources
2. Split documents into chunks
3. Create embeddings
4. Build vector stores
5. Perform similarity search
6. Build complete RAG chains
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def example_1_document_loading():
    """Example 1: Loading documents from different sources"""
    print("\n" + "="*60)
    print("Example 1: Document Loading")
    print("="*60)
    
    # Create a sample text file for demonstration
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that are context-aware and reason-based.
    LangChain provides modular components and chains for building complex applications.
    You can build chatbots, agents, RAG systems, and more with LangChain.
    The framework supports multiple LLM providers including OpenAI, Anthropic, and others.
    """
    
    with open("sample_document.txt", "w") as f:
        f.write(sample_text)
    
    # Load the document
    loader = TextLoader("sample_document.txt")
    documents = loader.load()
    
    print(f"Loaded {len(documents)} document(s)")
    print(f"Document content preview: {documents[0].page_content[:100]}...")
    print(f"Metadata: {documents[0].metadata}")
    
    # Cleanup
    os.remove("sample_document.txt")


def example_2_text_splitting():
    """Example 2: Splitting documents into chunks"""
    print("\n" + "="*60)
    print("Example 2: Text Splitting")
    print("="*60)
    
    # Sample long text
    long_text = """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills associated with the human mind, such as "learning" and "problem-solving".
    Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    Some popular accounts use the term "artificial intelligence" to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
    """ * 3  # Repeat to make it longer
    
    # Recursive character splitter (recommended)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )
    
    chunks = recursive_splitter.split_text(long_text)
    print(f"Recursive splitter created {len(chunks)} chunks")
    print(f"First chunk: {chunks[0][:80]}...")
    
    # Character splitter
    character_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
    )
    
    chunks_char = character_splitter.split_text(long_text)
    print(f"\nCharacter splitter created {len(chunks_char)} chunks")


def example_3_embeddings():
    """Example 3: Creating embeddings"""
    print("\n" + "="*60)
    print("Example 3: Embeddings")
    print("="*60)
    
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    texts = [
        "LangChain is a framework for LLM applications",
        "Python is a programming language",
        "Machine learning uses data to make predictions"
    ]
    
    # Generate embeddings
    text_embeddings = embeddings.embed_documents(texts)
    
    print(f"Generated {len(text_embeddings)} embeddings")
    print(f"Embedding dimension: {len(text_embeddings[0])}")
    print(f"First embedding preview: {text_embeddings[0][:5]}...")


def example_4_vector_store():
    """Example 4: Building and querying a vector store"""
    print("\n" + "="*60)
    print("Example 4: Vector Store")
    print("="*60)
    
    # Sample documents
    documents = [
        "LangChain is a framework for developing applications with language models.",
        "Python is a popular programming language for AI and data science.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are the foundation of deep learning.",
        "Vector databases store embeddings for similarity search."
    ]
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a simple in-memory vector store using Chroma
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        collection_name="demo_collection"
    )
    
    # Perform similarity search
    query = "What is LangChain?"
    results = vectorstore.similarity_search(query, k=2)
    
    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")
    
    # Cleanup
    vectorstore.delete_collection()


def example_5_simple_rag():
    """Example 5: Simple RAG pipeline"""
    print("\n" + "="*60)
    print("Example 5: Simple RAG Pipeline")
    print("="*60)
    
    # Sample knowledge base
    knowledge_base = [
        "LangChain is an open-source framework for building applications with large language models.",
        "It provides tools for prompt management, memory, and agent capabilities.",
        "LangChain supports multiple LLM providers including OpenAI, Anthropic, and Hugging Face.",
        "The framework uses chains to sequence multiple components together.",
        "RAG (Retrieval-Augmented Generation) combines retrieval with generation for better answers."
    ]
    
    # Create vector store
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = Chroma.from_texts(
        texts=knowledge_base,
        embedding=embeddings,
        collection_name="rag_demo"
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Create RAG chain
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the following context:
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Test the RAG chain
    question = "What is LangChain and what does it support?"
    answer = rag_chain.invoke(question)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Cleanup
    vectorstore.delete_collection()


def example_6_rag_with_sources():
    """Example 6: RAG with source citations"""
    print("\n" + "="*60)
    print("Example 6: RAG with Source Citations")
    print("="*60)
    
    # Documents with metadata
    docs_with_metadata = [
        {"page_content": "LangChain was created by Harrison Chase.", "metadata": {"source": "about"}},
        {"page_content": "The first version was released in 2022.", "metadata": {"source": "history"}},
        {"page_content": "LangChain is written in Python and JavaScript.", "metadata": {"source": "tech"}},
    ]
    
    # Create vector store with metadata
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = Chroma.from_documents(
        documents=docs_with_metadata,
        embedding=embeddings,
        collection_name="rag_with_sources"
    )
    
    # Search with scores
    results = vectorstore.similarity_search_with_score("Who created LangChain?", k=2)
    
    print("Search results with scores:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     Content: {doc.page_content}")
        print(f"     Source: {doc.metadata.get('source', 'unknown')}")
    
    # Cleanup
    vectorstore.delete_collection()


def example_7_document_qa():
    """Example 7: Document QA with a complete workflow"""
    print("\n" + "="*60)
    print("Example 7: Complete Document QA Workflow")
    print("="*60)
    
    # Create sample document
    sample_doc = """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
    
    Supervised learning uses labeled data to train models. Examples include classification and regression tasks.
    Unsupervised learning finds patterns in unlabeled data. Examples include clustering and dimensionality reduction.
    Reinforcement learning learns through trial and error by receiving rewards or penalties.
    
    Popular machine learning algorithms include linear regression, decision trees, random forests, and neural networks.
    Deep learning is a subset of machine learning that uses multi-layered neural networks.
    """
    
    with open("ml_intro.txt", "w") as f:
        f.write(sample_doc)
    
    # Load document
    loader = TextLoader("ml_intro.txt")
    documents = loader.load()
    
    # Split document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    
    print(f"Document split into {len(splits)} chunks")
    
    # Create vector store
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="document_qa"
    )
    
    # Create QA chain
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    retriever = vectorstore.as_retriever()
    
    prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question:
    
    Context: {context}
    
    Question: {question}
    
    If the answer is not in the context, say "I don't have enough information to answer this."
    
    Answer:
    """)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Ask questions
    questions = [
        "What are the three types of machine learning?",
        "What is supervised learning?"
    ]
    
    for question in questions:
        answer = qa_chain.invoke(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
    
    # Cleanup
    vectorstore.delete_collection()
    os.remove("ml_intro.txt")


def run_all_examples():
    """Run all examples in this module"""
    examples = [
        example_1_document_loading,
        example_2_text_splitting,
        example_3_embeddings,
        example_4_vector_store,
        example_5_simple_rag,
        example_6_rag_with_sources,
        example_7_document_qa
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {str(e)}")
            print("Note: Make sure you have OPENAI_API_KEY set in .env file")


if __name__ == "__main__":
    run_all_examples()

"""
ChainItUp - Web UI with Streamlit

A visual, interactive interface to learn LangChain through examples.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page configuration
st.set_page_config(
    page_title="ChainItUp - LangChain Learning",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)


def check_api_keys():
    """Check if API keys are configured"""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        return False, "No API keys found"
    return True, "API keys configured"


def basic_llm_demo():
    """Basic LLM demo"""
    st.markdown('<div class="feature-card"><h3>🤖 Basic LLM Demo</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        model = st.selectbox("Select Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        prompt = st.text_area("Enter your prompt", height=100, value="What is LangChain? Explain in one sentence.")
        
        if st.button("Generate Response", type="primary"):
            with col2:
                with st.spinner("Generating response..."):
                    try:
                        llm = ChatOpenAI(
                            model=model,
                            temperature=temperature,
                            api_key=os.getenv("OPENAI_API_KEY")
                        )
                        response = llm.invoke(prompt)
                        
                        st.markdown('<div class="success-box"><strong>Response:</strong></div>', unsafe_allow_html=True)
                        st.write(response.content)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


def prompt_template_demo():
    """Prompt template demo"""
    st.markdown('<div class="feature-card"><h3>📝 Prompt Template Demo</h3></div>', unsafe_allow_html=True)
    
    template_type = st.selectbox("Template Type", ["Simple Template", "Chat Template"])
    
    if template_type == "Simple Template":
        topic = st.text_input("Topic", "machine learning")
        audience = st.selectbox("Target Audience", ["beginner", "intermediate", "expert"])
        
        if st.button("Generate Prompt", type="primary"):
            from langchain_core.prompts import PromptTemplate
            
            template = PromptTemplate(
                template="Explain {topic} in simple terms for a {audience}.",
                input_variables=["topic", "audience"]
            )
            
            prompt = template.format(topic=topic, audience=audience)
            st.markdown('<div class="info-box"><strong>Generated Prompt:</strong></div>', unsafe_allow_html=True)
            st.code(prompt, language="text")
            
            # Execute the prompt
            with st.spinner("Getting response..."):
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
                response = llm.invoke(prompt)
                st.markdown('<div class="success-box"><strong>Response:</strong></div>', unsafe_allow_html=True)
                st.write(response.content)
    
    else:
        role = st.selectbox("Assistant Role", ["coding", "writing", "math"])
        task = st.text_input("Task", "help write a Python function")
        
        if st.button("Generate Response", type="primary"):
            from langchain_core.prompts import ChatPromptTemplate
            
            template = ChatPromptTemplate.from_messages([
                ("system", f"You are a helpful {role} assistant."),
                ("human", "Help me with {task}.")
            ])
            
            messages = template.format_messages(role=role, task=task)
            
            with st.spinner("Getting response..."):
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
                response = llm.invoke(messages)
                st.markdown('<div class="success-box"><strong>Response:</strong></div>', unsafe_allow_html=True)
                st.write(response.content)


def chain_demo():
    """Chain demo using LCEL"""
    st.markdown('<div class="feature-card"><h3>🔗 Chain Demo (LCEL)</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        chain_type = st.selectbox("Chain Type", ["Simple Chain", "Sequential Chain"])
        
        if chain_type == "Simple Chain":
            topic = st.text_input("Topic", "programming")
            
            if st.button("Run Chain", type="primary"):
                with col2:
                    with st.spinner("Running chain..."):
                        try:
                            llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
                            prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}.")
                            output_parser = StrOutputParser()
                            
                            chain = prompt | llm | output_parser
                            result = chain.invoke({"topic": topic})
                            
                            st.markdown('<div class="success-box"><strong>Chain Output:</strong></div>', unsafe_allow_html=True)
                            st.write(result)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        else:
            topic = st.text_input("Topic", "a robot")
            
            if st.button("Run Sequential Chain", type="primary"):
                with col2:
                    with st.spinner("Running sequential chain..."):
                        try:
                            llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
                            
                            story_prompt = ChatPromptTemplate.from_template(
                                "Write a one-sentence story about {topic}."
                            )
                            story_chain = story_prompt | llm | StrOutputParser()
                            
                            summary_prompt = ChatPromptTemplate.from_template(
                                "Summarize this story in 3 words: {story}"
                            )
                            summary_chain = summary_prompt | llm | StrOutputParser()
                            
                            from langchain_core.runnables import RunnablePassthrough
                            full_chain = {
                                "story": story_chain
                            } | RunnablePassthrough.assign(
                                summary=summary_chain
                            )
                            
                            result = full_chain.invoke({"topic": topic})
                            
                            st.markdown('<div class="info-box"><strong>Story:</strong></div>', unsafe_allow_html=True)
                            st.write(result["story"])
                            st.markdown('<div class="info-box"><strong>Summary:</strong></div>', unsafe_allow_html=True)
                            st.write(result["summary"])
                        except Exception as e:
                            st.error(f"Error: {str(e)}")


def memory_demo():
    """Memory demo"""
    st.markdown('<div class="feature-card"><h3>🧠 Memory Demo</h3></div>', unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        # Get response
        with st.spinner("Thinking..."):
            try:
                from langchain_core.messages import HumanMessage, AIMessage
                
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
                
                # Build message history
                messages = []
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
                
                response = llm.invoke(messages)
                assistant_message = response.content
                
                # Add assistant message to history
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
                st.chat_message("assistant").write(assistant_message)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()


def rag_demo():
    """Simple RAG demo"""
    st.markdown('<div class="feature-card"><h3>📚 RAG Demo</h3></div>', unsafe_allow_html=True)
    
    st.info("This demo creates a simple knowledge base and answers questions about it.")
    
    # Sample knowledge base
    knowledge_base = [
        "LangChain is an open-source framework for building applications with large language models.",
        "It provides tools for prompt management, memory, and agent capabilities.",
        "LangChain supports multiple LLM providers including OpenAI, Anthropic, and Hugging Face.",
        "The framework uses chains to sequence multiple components together.",
        "RAG (Retrieval-Augmented Generation) combines retrieval with generation for better answers."
    ]
    
    with st.expander("View Knowledge Base", expanded=False):
        for i, fact in enumerate(knowledge_base, 1):
            st.write(f"{i}. {fact}")
    
    question = st.text_input("Ask a question about LangChain:", "What is LangChain?")
    
    if st.button("Get Answer", type="primary"):
        with st.spinner("Searching knowledge base..."):
            try:
                from langchain_openai import OpenAIEmbeddings
                from langchain_chroma import Chroma
                from langchain_core.runnables import RunnablePassthrough
                
                embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
                
                # Create vector store
                vectorstore = Chroma.from_texts(
                    texts=knowledge_base,
                    embedding=embeddings,
                    collection_name="rag_demo"
                )
                
                retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                
                # Get relevant documents
                docs = retriever.invoke(question)
                
                st.markdown('<div class="info-box"><strong>Retrieved Context:</strong></div>', unsafe_allow_html=True)
                for i, doc in enumerate(docs, 1):
                    st.write(f"{i}. {doc.page_content}")
                
                # Generate answer
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
                
                answer = rag_chain.invoke(question)
                
                st.markdown('<div class="success-box"><strong>Answer:</strong></div>', unsafe_allow_html=True)
                st.write(answer)
                
                # Cleanup
                vectorstore.delete_collection()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")


def main():
    """Main application"""
    # Header
    st.markdown('<div class="main-header">🔗 ChainItUp</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive LangChain Learning Platform</div>', unsafe_allow_html=True)
    
    # Check API keys
    api_keys_ok, api_status = check_api_keys()
    
    if not api_keys_ok:
        st.error("⚠️ No API keys found! Please add OPENAI_API_KEY to your .env file")
        st.info("Copy .env.example to .env and add your API key")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Features")
        
        page = st.radio(
            "Select a Demo",
            ["🏠 Home", "🤖 Basic LLM", "📝 Prompt Templates", "🔗 Chains", "🧠 Memory", "📚 RAG"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("### 📖 About")
        st.markdown("""
        This app demonstrates LangChain features through interactive examples.
        
        Each section shows a different aspect of LangChain:
        - **Basic LLM**: Model initialization and usage
        - **Prompt Templates**: Dynamic prompt engineering
        - **Chains**: Building workflows with LCEL
        - **Memory**: Conversation history management
        - **RAG**: Document Q&A systems
        """)
        
        st.divider()
        
        st.markdown("### 🔑 API Status")
        if api_keys_ok:
            st.success("✅ API keys configured")
        else:
            st.error("❌ API keys missing")
    
    # Main content
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to ChainItUp! 🚀
        
        This interactive application helps you learn LangChain through hands-on demonstrations.
        
        ### Getting Started
        
        1. **Select a demo** from the sidebar
        2. **Follow the instructions** in each section
        3. **Experiment** with different parameters
        4. **Learn** by seeing real LangChain code in action
        
        ### Available Demos
        
        - **🤖 Basic LLM**: Learn to initialize and use different LLM providers
        - **📝 Prompt Templates**: Master dynamic prompt engineering
        - **🔗 Chains**: Build complex workflows with LCEL
        - **🧠 Memory**: Handle conversation history
        - **📚 RAG**: Build document Q&A systems
        
        ### Tips
        
        - Adjust the **temperature** slider to control creativity
        - Try different **models** to see performance differences
        - Use the **chat interface** in the Memory demo to see context retention
        - Explore the **RAG demo** to understand retrieval-augmented generation
        
        Happy learning! 🎓
        """)
    
    elif page == "🤖 Basic LLM":
        basic_llm_demo()
    
    elif page == "📝 Prompt Templates":
        prompt_template_demo()
    
    elif page == "🔗 Chains":
        chain_demo()
    
    elif page == "🧠 Memory":
        memory_demo()
    
    elif page == "📚 RAG":
        rag_demo()


if __name__ == "__main__":
    main()

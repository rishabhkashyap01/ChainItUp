"""
Memory - Managing Conversation History

This module demonstrates different memory types in LangChain.
Learn how to:
1. Use ConversationBufferMemory
2. Use ConversationBufferWindowMemory
3. Use ConversationSummaryMemory
4. Use ConversationKGMemory (Knowledge Graph)
5. Use custom memory configurations
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationKGMemory
)


def example_1_buffer_memory():
    """Example 1: ConversationBufferMemory - stores all messages"""
    print("\n" + "="*60)
    print("Example 1: Conversation Buffer Memory")
    print("="*60)
    
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )
    
    # Add some conversation history
    memory.save_context(
        {"input": "Hi, I'm learning LangChain."},
        {"output": "That's great! LangChain is a powerful framework."}
    )
    memory.save_context(
        {"input": "What can I build with it?"},
        {"output": "You can build chatbots, agents, RAG systems, and more!"}
    )
    
    # Retrieve memory
    history = memory.load_memory_variables({})
    print(f"Conversation History ({len(history['chat_history'])} messages):")
    for msg in history['chat_history']:
        print(f"  {msg.type}: {msg.content[:50]}...")


def example_2_window_memory():
    """Example 2: ConversationBufferWindowMemory - keeps last k messages"""
    print("\n" + "="*60)
    print("Example 2: Conversation Window Memory")
    print("="*60)
    
    memory = ConversationBufferWindowMemory(
        k=2,  # Keep only last 2 exchanges
        return_messages=True,
        memory_key="chat_history"
    )
    
    # Add conversation history
    messages = [
        ("Hi!", "Hello!"),
        ("How are you?", "I'm doing well!"),
        ("What's your name?", "I'm an AI assistant."),
        ("What can you do?", "I can help with various tasks.")
    ]
    
    for user_msg, ai_msg in messages:
        memory.save_context({"input": user_msg}, {"output": ai_msg})
    
    history = memory.load_memory_variables({})
    print(f"Window Memory (last 2 exchanges, {len(history['chat_history'])} messages):")
    for msg in history['chat_history']:
        print(f"  {msg.type}: {msg.content}")


def example_3_summary_memory():
    """Example 3: ConversationSummaryMemory - summarizes conversation"""
    print("\n" + "="*60)
    print("Example 3: Conversation Summary Memory")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    memory = ConversationSummaryMemory(
        llm=llm,
        return_messages=False,
        memory_key="chat_summary"
    )
    
    # Add conversation
    memory.save_context(
        {"input": "I'm interested in machine learning."},
        {"output": "Machine learning is a subset of AI focused on algorithms."}
    )
    memory.save_context(
        {"input": "What about deep learning?"},
        {"output": "Deep learning uses neural networks with multiple layers."}
    )
    memory.save_context(
        {"input": "How are they related?"},
        {"output": "Deep learning is a subset of machine learning."}
    )
    
    summary = memory.load_memory_variables({})
    print(f"Conversation Summary:")
    print(f"  {summary['chat_summary']}")


def example_4_kg_memory():
    """Example 4: ConversationKGMemory - extracts knowledge graph"""
    print("\n" + "="*60)
    print("Example 4: Knowledge Graph Memory")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    memory = ConversationKGMemory(llm=llm)
    
    # Add conversation with entities and relationships
    memory.save_context(
        {"input": "Elon Musk founded SpaceX in 2002."},
        {"output": "That's correct. SpaceX is a private aerospace manufacturer."}
    )
    memory.save_context(
        {"input": "Who founded Tesla?"},
        {"output": "Tesla was founded by Martin Eberhard and Marc Tarpenning."}
    )
    memory.save_context(
        {"input": "Elon Musk joined Tesla later."},
        {"output": "Yes, Elon Musk joined Tesla in 2004 and became CEO."}
    )
    
    kg = memory.load_memory_variables({})
    print(f"Knowledge Graph:")
    print(f"  {kg.get('knowledge_graph', 'No graph extracted')}")


def example_5_memory_with_chain():
    """Example 5: Using memory with a simple chain"""
    print("\n" + "="*60)
    print("Example 5: Memory with Chain")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationBufferMemory(return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Simulate a conversation
    conversation = [
        "My name is Bob.",
        "Nice to meet you, Bob!",
        "What's my name?",
        "I work as a software engineer.",
        "What do I do for work?"
    ]
    
    for i, user_input in enumerate(conversation):
        if i % 2 == 0:  # User input
            # Get relevant history
            history = memory.load_memory_variables({})["history"]
            
            # Create prompt with history
            messages = prompt.format_messages(
                history=history,
                input=user_input
            )
            
            # Get response
            response = llm.invoke(messages)
            ai_output = response.content
            
            print(f"User: {user_input}")
            print(f"AI: {ai_output}\n")
            
            # Save to memory
            memory.save_context({"input": user_input}, {"output": ai_output})


def example_6_memory_types_comparison():
    """Example 6: Compare different memory types"""
    print("\n" + "="*60)
    print("Example 6: Memory Types Comparison")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create different memory types
    buffer_memory = ConversationBufferMemory(return_messages=True)
    window_memory = ConversationBufferWindowMemory(k=1, return_messages=True)
    
    # Add same conversation to both
    for i in range(5):
        buffer_memory.save_context(
            {"input": f"Message {i+1}"},
            {"output": f"Response {i+1}"}
        )
        window_memory.save_context(
            {"input": f"Message {i+1}"},
            {"output": f"Response {i+1}"}
        )
    
    buffer_history = buffer_memory.load_memory_variables({})
    window_history = window_memory.load_memory_variables({})
    
    print(f"Buffer Memory: {len(buffer_history['history'])} messages stored")
    print(f"Window Memory (k=1): {len(window_history['history'])} messages stored")


def run_all_examples():
    """Run all examples in this module"""
    examples = [
        example_1_buffer_memory,
        example_2_window_memory,
        example_3_summary_memory,
        example_4_kg_memory,
        example_5_memory_with_chain,
        example_6_memory_types_comparison
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {str(e)}")


if __name__ == "__main__":
    run_all_examples()

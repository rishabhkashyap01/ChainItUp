"""
Basic LLM Usage - LangChain Fundamentals

This module demonstrates the most basic usage of LangChain with different LLM providers.
Learn how to:
1. Initialize different LLM providers (OpenAI, Anthropic)
2. Make simple calls to LLMs
3. Use streaming responses
4. Handle different parameters (temperature, max_tokens)
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage


def example_1_simple_openai_call():
    """Example 1: Simple call to OpenAI's GPT model"""
    print("\n" + "="*60)
    print("Example 1: Simple OpenAI Call")
    print("="*60)
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    response = llm.invoke("What is LangChain? Explain in one sentence.")
    print(f"Response: {response.content}")
    print(f"Model: {response.response_metadata.get('model', 'unknown')}")


def example_2_simple_anthropic_call():
    """Example 2: Simple call to Anthropic's Claude model"""
    print("\n" + "="*60)
    print("Example 2: Simple Anthropic Call")
    print("="*60)
    
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    response = llm.invoke("What is LangChain? Explain in one sentence.")
    print(f"Response: {response.content}")


def example_3_message_history():
    """Example 3: Using message history with system and human messages"""
    print("\n" + "="*60)
    print("Example 3: Message History")
    print("="*60)
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    messages = [
        SystemMessage(content="You are a helpful AI assistant that explains technical concepts simply."),
        HumanMessage(content="Explain what a vector database is.")
    ]
    
    response = llm.invoke(messages)
    print(f"Response: {response.content}")


def example_4_streaming_response():
    """Example 4: Streaming responses from the LLM"""
    print("\n" + "="*60)
    print("Example 4: Streaming Response")
    print("="*60)
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print("Streaming response: ", end="", flush=True)
    for chunk in llm.stream("Count from 1 to 5"):
        print(chunk.content, end="", flush=True)
    print()


def example_5_batch_processing():
    """Example 5: Processing multiple prompts in batch"""
    print("\n" + "="*60)
    print("Example 5: Batch Processing")
    print("="*60)
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ]
    
    responses = llm.batch(prompts)
    for i, response in enumerate(responses, 1):
        print(f"\nPrompt {i}: {prompts[i-1]}")
        print(f"Response: {response.content[:100]}...")


def example_6_temperature_control():
    """Example 6: Understanding temperature parameter"""
    print("\n" + "="*60)
    print("Example 6: Temperature Control")
    print("="*60)
    
    prompt = "Write a creative opening sentence for a sci-fi novel."
    
    # Low temperature = more deterministic
    llm_low = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # High temperature = more creative/random
    llm_high = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.9,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print("Temperature 0.1 (Low):")
    print(llm_low.invoke(prompt).content)
    
    print("\nTemperature 0.9 (High):")
    print(llm_high.invoke(prompt).content)


def run_all_examples():
    """Run all examples in this module"""
    examples = [
        example_1_simple_openai_call,
        example_2_simple_anthropic_call,
        example_3_message_history,
        example_4_streaming_response,
        example_5_batch_processing,
        example_6_temperature_control
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {str(e)}")
            print("Make sure you have the required API keys set in .env file")


if __name__ == "__main__":
    run_all_examples()

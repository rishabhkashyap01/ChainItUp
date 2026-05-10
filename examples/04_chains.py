"""
Chains - Building Complex Workflows

This module demonstrates different chain types in LangChain.
Learn how to:
1. Create simple LLM chains
2. Build sequential chains
3. Use LangChain Expression Language (LCEL)
4. Create custom chains
5. Chain multiple components together
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def example_1_simple_chain():
    """Example 1: Simple LLM Chain using LCEL"""
    print("\n" + "="*60)
    print("Example 1: Simple LLM Chain (LCEL)")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}.")
    output_parser = StrOutputParser()
    
    # Create chain using LCEL (|)
    chain = prompt | llm | output_parser
    
    result = chain.invoke({"topic": "programming"})
    print(f"Result: {result}")


def example_2_sequential_chain():
    """Example 2: Sequential chain - output of one becomes input of next"""
    print("\n" + "="*60)
    print("Example 2: Sequential Chain")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    # First chain: Generate a story
    story_prompt = ChatPromptTemplate.from_template(
        "Write a one-sentence story about {topic}."
    )
    story_chain = story_prompt | llm | StrOutputParser()
    
    # Second chain: Summarize the story
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this story in 3 words: {story}"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    # Combine chains
    full_chain = {
        "story": story_chain
    } | RunnablePassthrough.assign(
        summary=summary_chain
    )
    
    result = full_chain.invoke({"topic": "a robot"})
    print(f"Story: {result['story']}")
    print(f"Summary: {result['summary']}")


def example_3_runnable_passthrough():
    """Example 3: Using RunnablePassthrough to pass data through"""
    print("\n" + "="*60)
    print("Example 3: RunnablePassthrough")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template(
        "Translate '{text}' to {language}."
    )
    
    chain = (
        RunnablePassthrough.assign(
            translated=prompt | llm | StrOutputParser()
        )
    )
    
    result = chain.invoke({
        "text": "Hello, how are you?",
        "language": "Spanish"
    })
    
    print(f"Original: {result['text']}")
    print(f"Language: {result['language']}")
    print(f"Translated: {result['translated']}")


def example_4_runnable_lambda():
    """Example 4: Using RunnableLambda for custom transformations"""
    print("\n" + "="*60)
    print("Example 4: RunnableLambda")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    # Custom function to preprocess input
    def preprocess(text):
        return f"Please explain this simply: {text}"
    
    # Custom function to postprocess output
    def postprocess(text):
        return f"EXPLANATION: {text}"
    
    chain = (
        RunnableLambda(preprocess)
        | ChatPromptTemplate.from_template("{text}")
        | llm
        | StrOutputParser()
        | RunnableLambda(postprocess)
    )
    
    result = chain.invoke({"text": "quantum computing"})
    print(f"Result: {result}")


def example_5_branching_chain():
    """Example 5: Branching chain - conditional logic"""
    print("\n" + "="*60)
    print("Example 5: Branching Chain")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    # Define different prompts for different conditions
    formal_prompt = ChatPromptTemplate.from_template(
        "Write a formal response to: {input}"
    )
    casual_prompt = ChatPromptTemplate.from_template(
        "Write a casual response to: {input}"
    )
    
    # Function to determine which chain to use
    def route(input_dict):
        tone = input_dict.get("tone", "formal")
        if tone == "casual":
            return casual_prompt | llm | StrOutputParser()
        else:
            return formal_prompt | llm | StrOutputParser()
    
    chain = RunnableLambda(route)
    
    # Test with different tones
    formal_result = chain.invoke({"input": "Hello!", "tone": "formal"})
    casual_result = chain.invoke({"input": "Hello!", "tone": "casual"})
    
    print(f"Formal: {formal_result}")
    print(f"Casual: {casual_result}")


def example_6_parallel_chain():
    """Example 6: Running chains in parallel"""
    print("\n" + "="*60)
    print("Example 6: Parallel Chain")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    # Define parallel tasks
    joke_prompt = ChatPromptTemplate.from_template("Tell a joke about {topic}.")
    fact_prompt = ChatPromptTemplate.from_template("Tell a fact about {topic}.")
    
    joke_chain = joke_prompt | llm | StrOutputParser()
    fact_chain = fact_prompt | llm | StrOutputParser()
    
    # Run in parallel
    from langchain_core.runnables import RunnableParallel
    
    chain = RunnableParallel(
        joke=joke_chain,
        fact=fact_chain
    )
    
    result = chain.invoke({"topic": "cats"})
    print(f"Joke: {result['joke']}")
    print(f"Fact: {result['fact']}")


def example_7_chain_with_memory():
    """Example 7: Chain with conversation memory"""
    print("\n" + "="*60)
    print("Example 7: Chain with Memory")
    print("="*60)
    
    from langchain.memory import ConversationBufferMemory
    from langchain_core.prompts import MessagesPlaceholder
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationBufferMemory(return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = (
        RunnablePassthrough.assign(
            history=lambda x: memory.load_memory_variables({})["history"]
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # First interaction
    result1 = chain.invoke({"input": "My name is Alice."})
    memory.save_context({"input": "My name is Alice."}, {"output": result1})
    
    # Second interaction
    result2 = chain.invoke({"input": "What's my name?"})
    
    print(f"First response: {result1}")
    print(f"Second response: {result2}")


def example_8_streaming_chain():
    """Example 8: Streaming through a chain"""
    print("\n" + "="*60)
    print("Example 8: Streaming Chain")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template("Count from 1 to {num}.")
    chain = prompt | llm | StrOutputParser()
    
    print("Streaming output: ", end="", flush=True)
    for chunk in chain.stream({"num": 5}):
        print(chunk, end="", flush=True)
    print()


def run_all_examples():
    """Run all examples in this module"""
    examples = [
        example_1_simple_chain,
        example_2_sequential_chain,
        example_3_runnable_passthrough,
        example_4_runnable_lambda,
        example_5_branching_chain,
        example_6_parallel_chain,
        example_7_chain_with_memory,
        example_8_streaming_chain
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {str(e)}")


if __name__ == "__main__":
    run_all_examples()

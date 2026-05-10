"""
Prompt Templates - Dynamic Prompt Engineering

This module demonstrates how to use prompt templates in LangChain.
Learn how to:
1. Create basic prompt templates
2. Use ChatPromptTemplate for structured conversations
3. Partial templates with pre-filled values
4. Compose multiple templates together
5. Use message prompt templates
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage


def example_1_basic_prompt_template():
    """Example 1: Basic PromptTemplate with variables"""
    print("\n" + "="*60)
    print("Example 1: Basic Prompt Template")
    print("="*60)
    
    template = PromptTemplate(
        template="Explain {topic} in simple terms for a {audience}.",
        input_variables=["topic", "audience"]
    )
    
    prompt = template.format(topic="machine learning", audience="beginner")
    print(f"Generated Prompt: {prompt}")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(prompt)
    print(f"Response: {response.content}")


def example_2_chat_prompt_template():
    """Example 2: ChatPromptTemplate for structured conversations"""
    print("\n" + "="*60)
    print("Example 2: Chat Prompt Template")
    print("="*60)
    
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful {role} assistant."),
        ("human", "Help me with {task}.")
    ])
    
    prompt = template.format_messages(role="coding", task="writing a Python function")
    print(f"Generated Messages: {len(prompt)} messages")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(prompt)
    print(f"Response: {response.content[:200]}...")


def example_3_partial_templates():
    """Example 3: Partial templates with pre-filled values"""
    print("\n" + "="*60)
    print("Example 3: Partial Templates")
    print("="*60)
    
    # Create a partial template with some values pre-filled
    template = PromptTemplate(
        template="Write a {adjective} story about {topic}. Keep it under {max_words} words.",
        input_variables=["topic", "max_words"],
        partial_variables={"adjective": "funny"}
    )
    
    prompt = template.format(topic="a cat learning to code", max_words=50)
    print(f"Generated Prompt: {prompt}")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(prompt)
    print(f"Response: {response.content}")


def example_4_message_placeholder():
    """Example 4: Using MessagesPlaceholder for conversation history"""
    print("\n" + "="*60)
    print("Example 4: Message Placeholder for History")
    print("="*60)
    
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Simulate conversation history
    history = [
        HumanMessage(content="My name is Alice."),
        HumanMessage(content="I like programming.")
    ]
    
    prompt = template.format_messages(
        history=history,
        input="What's my name?"
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(prompt)
    print(f"Response: {response.content}")


def example_5_template_composition():
    """Example 5: Composing multiple templates"""
    print("\n" + "="*60)
    print("Example 5: Template Composition")
    print("="*60)
    
    system_template = PromptTemplate(
        template="You are an expert in {field}.",
        input_variables=["field"]
    )
    
    user_template = PromptTemplate(
        template="Answer this question: {question}",
        input_variables=["question"]
    )
    
    full_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", user_template)
    ])
    
    prompt = full_template.format_messages(field="astronomy", question="What is a black hole?")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(prompt)
    print(f"Response: {response.content[:200]}...")


def example_6_few_shot_prompting():
    """Example 6: Few-shot prompting with examples"""
    print("\n" + "="*60)
    print("Example 6: Few-Shot Prompting")
    print("="*60)
    
    examples = [
        {"input": "happy", "output": "joyful"},
        {"input": "sad", "output": "sorrowful"},
        {"input": "angry", "output": "furious"}
    ]
    
    example_template = PromptTemplate(
        template="Input: {input}\nOutput: {output}",
        input_variables=["input", "output"]
    )
    
    # Build few-shot prompt
    few_shot_prompt = "Here are some examples:\n\n"
    for example in examples:
        few_shot_prompt += example_template.format(**example) + "\n\n"
    
    few_shot_prompt += "Input: excited\nOutput:"
    
    print(f"Few-shot Prompt:\n{few_shot_prompt}")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(few_shot_prompt)
    print(f"Response: {response.content}")


def example_7_validation_template():
    """Example 7: Template with input validation"""
    print("\n" + "="*60)
    print("Example 7: Template with Validation")
    print("="*60)
    
    template = PromptTemplate(
        template="Summarize the following text in {num_sentences} sentences:\n\n{text}",
        input_variables=["text", "num_sentences"]
    )
    
    # Validate inputs
    text = "LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and reason-based. LangChain provides modular components and chains for building complex applications."
    num_sentences = "2"
    
    prompt = template.format(text=text, num_sentences=num_sentences)
    print(f"Generated Prompt (length: {len(prompt)} chars)")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(prompt)
    print(f"Response: {response.content}")


def run_all_examples():
    """Run all examples in this module"""
    examples = [
        example_1_basic_prompt_template,
        example_2_chat_prompt_template,
        example_3_partial_templates,
        example_4_message_placeholder,
        example_5_template_composition,
        example_6_few_shot_prompting,
        example_7_validation_template
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {str(e)}")


if __name__ == "__main__":
    run_all_examples()

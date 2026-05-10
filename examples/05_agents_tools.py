"""
Agents and Tools - Building Autonomous AI Systems

This module demonstrates agents and tools in LangChain.
Learn how to:
1. Create custom tools
2. Use built-in tools
3. Build ReAct agents
4. Use tool-calling agents
5. Create multi-agent systems
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun


@tool
def calculator(expression: str) -> str:
    """A simple calculator that evaluates mathematical expressions.
    
    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2")
    """
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def word_counter(text: str) -> str:
    """Count the number of words in a text.
    
    Args:
        text: The text to count words in
    """
    words = text.split()
    return f"The text contains {len(words)} words."


def example_1_custom_tools():
    """Example 1: Creating and using custom tools"""
    print("\n" + "="*60)
    print("Example 1: Custom Tools")
    print("="*60)
    
    # Test custom tools
    print(calculator.invoke("15 * 3"))
    print(get_current_time.invoke(""))
    print(word_counter.invoke("Hello world, this is a test."))


def example_2_builtin_tools():
    """Example 2: Using built-in tools"""
    print("\n" + "="*60)
    print("Example 2: Built-in Tools")
    print("="*60)
    
    # Use DuckDuckGo search
    search = DuckDuckGoSearchRun()
    
    result = search.invoke("What is LangChain?")
    print(f"Search result: {result[:200]}...")


def example_3_simple_agent():
    """Example 3: Creating a simple tool-calling agent"""
    print("\n" + "="*60)
    print("Example 3: Simple Tool-Calling Agent")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    tools = [calculator, get_current_time, word_counter]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    result = agent_executor.invoke({
        "input": "What is 25 * 4 and also count the words in this sentence."
    })
    
    print(f"\nFinal answer: {result['output']}")


def example_4_agent_with_search():
    """Example 4: Agent with search capability"""
    print("\n" + "="*60)
    print("Example 4: Agent with Search")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    search = DuckDuckGoSearchRun()
    tools = [search, calculator]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can search the web and calculate."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )
    
    result = agent_executor.invoke({
        "input": "Search for the current population of Tokyo and multiply it by 2."
    })
    
    print(f"Result: {result['output']}")


def example_5_multi_tool_agent():
    """Example 5: Agent with multiple specialized tools"""
    print("\n" + "="*60)
    print("Example 5: Multi-Tool Agent")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    @tool
    def uppercase(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()
    
    @tool
    def lowercase(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    @tool
    def reverse(text: str) -> str:
        """Reverse the text."""
        return text[::-1]
    
    tools = [calculator, uppercase, lowercase, reverse]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with various text manipulation and calculation tools."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )
    
    result = agent_executor.invoke({
        "input": "Convert 'hello world' to uppercase and then reverse it."
    })
    
    print(f"Result: {result['output']}")


def example_6_agent_with_memory():
    """Example 6: Agent with conversation memory"""
    print("\n" + "="*60)
    print("Example 6: Agent with Memory")
    print("="*60)
    
    from langchain.memory import ConversationBufferMemory
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    tools = [calculator, get_current_time]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with tools and memory."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True
    )
    
    # First interaction
    result1 = agent_executor.invoke({"input": "What is 10 + 5?"})
    print(f"First: {result1['output']}")
    
    # Second interaction (should remember)
    result2 = agent_executor.invoke({"input": "What was the previous result multiplied by 2?"})
    print(f"Second: {result2['output']}")


def example_7_tool_error_handling():
    """Example 7: Handling tool errors gracefully"""
    print("\n" + "="*60)
    print("Example 7: Tool Error Handling")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    @tool
    def division_tool(a: float, b: float) -> str:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return f"{a} / {b} = {a / b}"
    
    tools = [division_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Handle errors gracefully."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    result = agent_executor.invoke({
        "input": "Divide 10 by 0 and then tell me what happened."
    })
    
    print(f"Result: {result['output']}")


def run_all_examples():
    """Run all examples in this module"""
    examples = [
        example_1_custom_tools,
        example_2_builtin_tools,
        example_3_simple_agent,
        example_4_agent_with_search,
        example_5_multi_tool_agent,
        example_6_agent_with_memory,
        example_7_tool_error_handling
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {str(e)}")
            print("Note: Some examples may require additional setup or API keys")


if __name__ == "__main__":
    run_all_examples()

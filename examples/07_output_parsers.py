"""
Output Parsers - Structuring LLM Outputs

This module demonstrates output parsers in LangChain.
Learn how to:
1. Use StrOutputParser for string outputs
2. Use PydanticOutputParser for structured data
3. Use CommaSeparatedListOutputParser
4. Use JSON output parsers
5. Create custom output parsers
6. Handle parsing errors
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser
)
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field
from typing import List


def example_1_string_parser():
    """Example 1: Basic string output parser"""
    print("\n" + "="*60)
    print("Example 1: String Output Parser")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}.")
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"topic": "programming"})
    
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")


def example_2_pydantic_parser():
    """Example 2: Pydantic output parser for structured data"""
    print("\n" + "="*60)
    print("Example 2: Pydantic Output Parser")
    print("="*60)
    
    # Define the expected output structure
    class MovieReview(BaseModel):
        title: str = Field(description="The title of the movie")
        rating: float = Field(description="Rating from 1-10")
        summary: str = Field(description="Brief summary of the review")
        pros: List[str] = Field(description="List of positive aspects")
        cons: List[str] = Field(description="List of negative aspects")
    
    parser = PydanticOutputParser(pydantic_object=MovieReview)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template("""
    Write a review for the movie "The Matrix".
    
    {format_instructions}
    """)
    
    chain = prompt | llm | parser
    result = chain.invoke({"format_instructions": parser.get_format_instructions()})
    
    print(f"Review for: {result.title}")
    print(f"Rating: {result.rating}/10")
    print(f"Summary: {result.summary}")
    print(f"Pros: {', '.join(result.pros)}")
    print(f"Cons: {', '.join(result.cons)}")


def example_3_list_parser():
    """Example 3: Comma-separated list parser"""
    print("\n" + "="*60)
    print("Example 3: Comma-Separated List Parser")
    print("="*60)
    
    parser = CommaSeparatedListOutputParser()
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template("""
    List 5 programming languages.
    {format_instructions}
    """)
    
    chain = prompt | llm | parser
    result = chain.invoke({"format_instructions": parser.get_format_instructions()})
    
    print(f"Result type: {type(result)}")
    print(f"Languages: {result}")
    for i, lang in enumerate(result, 1):
        print(f"  {i}. {lang}")


def example_4_json_parser():
    """Example 4: JSON output parser"""
    print("\n" + "="*60)
    print("Example 4: JSON Output Parser")
    print("="*60)
    
    parser = JsonOutputParser()
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template("""
    Extract information from the following text:
    
    "John Doe is a 35-year-old software engineer living in San Francisco. 
    He works at Google and has 10 years of experience."
    
    Provide the output as JSON with keys: name, age, occupation, city, company, experience.
    
    {format_instructions}
    """)
    
    chain = prompt | llm | parser
    result = chain.invoke({"format_instructions": parser.get_format_instructions()})
    
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    print(f"\nExtracted information:")
    for key, value in result.items():
        print(f"  {key}: {value}")


def example_5_custom_parser():
    """Example 5: Custom output parser"""
    print("\n" + "="*60)
    print("Example 5: Custom Output Parser")
    print("="*60)
    
    from langchain_core.output_parsers import BaseOutputParser
    
    class UppercaseParser(BaseOutputParser):
        """Custom parser that converts output to uppercase"""
        
        def parse(self, text: str) -> str:
            return text.upper()
        
        @property
        def _type(self) -> str:
            return "uppercase_parser"
    
    parser = UppercaseParser()
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_template("Say hello in a friendly way.")
    
    chain = prompt | llm | StrOutputParser() | parser
    result = chain.invoke({})
    
    print(f"Original would be lowercase, but parser converts to uppercase:")
    print(f"Result: {result}")


def example_6_error_handling():
    """Example 6: Handling parsing errors"""
    print("\n" + "="*60)
    print("Example 6: Error Handling")
    print("="*60)
    
    class Person(BaseModel):
        name: str
        age: int
    
    parser = PydanticOutputParser(pydantic_object=Person)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template("""
    Extract name and age from: "Alice is twenty-five"
    
    {format_instructions}
    """)
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"format_instructions": parser.get_format_instructions()})
        print(f"Result: {result}")
    except OutputParserException as e:
        print(f"Parser error occurred: {e}")
        print("This happens when the LLM doesn't follow the format instructions exactly.")
        
        # Try with fix
        print("\nTrying with fix...")
        try:
            fixed = parser.parse_with_prompt(str(e), prompt)
            print(f"Fixed result: {fixed}")
        except Exception as e2:
            print(f"Could not fix: {e2}")


def example_7_multiple_parsers():
    """Example 7: Using multiple parsers in sequence"""
    print("\n" + "="*60)
    print("Example 7: Multiple Parsers in Sequence")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template(
        "List 3 fruits, 3 vegetables, and 3 colors."
    )
    
    # First get string, then parse as list
    list_parser = CommaSeparatedListOutputParser()
    
    chain = prompt | llm | StrOutputParser() | list_parser
    result = chain.invoke({})
    
    print(f"Total items: {len(result)}")
    print(f"Items: {result}")


def example_8_parser_with_validation():
    """Example 8: Parser with validation"""
    print("\n" + "="*60)
    print("Example 8: Parser with Validation")
    print("="*60)
    
    from pydantic import validator
    
    class Product(BaseModel):
        name: str
        price: float
        quantity: int = Field(ge=0, description="Quantity must be non-negative")
        
        @validator('price')
        def price_must_be_positive(cls, v):
            if v <= 0:
                raise ValueError('Price must be positive')
            return v
    
    parser = PydanticOutputParser(pydantic_object=Product)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template("""
    Extract product information from:
    "Laptop priced at $999.99, 5 units in stock"
    
    {format_instructions}
    """)
    
    chain = prompt | llm | parser
    result = chain.invoke({"format_instructions": parser.get_format_instructions()})
    
    print(f"Product: {result.name}")
    print(f"Price: ${result.price}")
    print(f"Quantity: {result.quantity}")
    print(f"Validation passed!")


def run_all_examples():
    """Run all examples in this module"""
    examples = [
        example_1_string_parser,
        example_2_pydantic_parser,
        example_3_list_parser,
        example_4_json_parser,
        example_5_custom_parser,
        example_6_error_handling,
        example_7_multiple_parsers,
        example_8_parser_with_validation
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {str(e)}")
            print("Note: Some examples may fail if the LLM doesn't follow format instructions")


if __name__ == "__main__":
    run_all_examples()

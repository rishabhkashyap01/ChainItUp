"""
ChainItUp - Interactive LangChain Learning Application

A comprehensive application demonstrating all major LangChain features
with an interactive menu for easy learning and exploration.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import example modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from examples import basic_llm
from examples import prompt_templates
from examples import memory
from examples import chains
from examples import agents_tools
from examples import rag
from examples import output_parsers


def print_header():
    """Print the application header"""
    print("\n" + "="*70)
    print(" "*15 + "ChainItUp - LangChain Learning App")
    print("="*70)
    print("\nLearn LangChain through interactive examples!")
    print("Each module demonstrates key features with clear explanations.\n")


def print_menu():
    """Print the main menu"""
    menu_options = [
        ("1", "Basic LLM Usage", "Learn to initialize and use different LLM providers"),
        ("2", "Prompt Templates", "Master dynamic prompt engineering"),
        ("3", "Memory Management", "Handle conversation history"),
        ("4", "Chains", "Build complex workflows with LCEL"),
        ("5", "Agents & Tools", "Create autonomous AI systems"),
        ("6", "RAG (Retrieval-Augmented Generation)", "Build document Q&A systems"),
        ("7", "Output Parsers", "Structure LLM outputs"),
        ("8", "Run All Examples", "Execute all modules sequentially"),
        ("0", "Exit", "Quit the application")
    ]
    
    print("Main Menu:")
    print("-" * 70)
    for num, title, description in menu_options:
        print(f"  [{num}] {title:<40} - {description}")
    print("-" * 70)


def run_module(module_name, module):
    """Run a specific example module"""
    print(f"\n{'='*70}")
    print(f"Running: {module_name}")
    print(f"{'='*70}\n")
    
    try:
        module.run_all_examples()
        print(f"\n{'='*70}")
        print(f"✓ {module_name} completed successfully!")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ Error running {module_name}: {str(e)}")
        print(f"{'='*70}\n")
        print("Tip: Make sure you have the required API keys in your .env file")


def check_api_keys():
    """Check if API keys are configured"""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("\n⚠️  Warning: No API keys found!")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file")
        print("Copy .env.example to .env and add your API keys.\n")
        return False
    
    if not openai_key:
        print("\n⚠️  Note: OPENAI_API_KEY not set. Some examples may fail.")
        print("Consider adding it to your .env file for full functionality.\n")
    
    if not anthropic_key:
        print("\n⚠️  Note: ANTHROPIC_API_KEY not set. Anthropic examples will be skipped.\n")
    
    return True


def main():
    """Main application loop"""
    print_header()
    
    # Check API keys
    if not check_api_keys():
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (0-8): ").strip()
        
        if choice == "0":
            print("\nThank you for using ChainItUp!")
            print("Happy learning! 🚀\n")
            break
        
        elif choice == "1":
            run_module("Basic LLM Usage", basic_llm)
        
        elif choice == "2":
            run_module("Prompt Templates", prompt_templates)
        
        elif choice == "3":
            run_module("Memory Management", memory)
        
        elif choice == "4":
            run_module("Chains", chains)
        
        elif choice == "5":
            run_module("Agents & Tools", agents_tools)
        
        elif choice == "6":
            run_module("RAG (Retrieval-Augmented Generation)", rag)
        
        elif choice == "7":
            run_module("Output Parsers", output_parsers)
        
        elif choice == "8":
            print("\n" + "="*70)
            print("Running All Examples")
            print("="*70 + "\n")
            
            modules = [
                ("Basic LLM Usage", basic_llm),
                ("Prompt Templates", prompt_templates),
                ("Memory Management", memory),
                ("Chains", chains),
                ("Agents & Tools", agents_tools),
                ("RAG", rag),
                ("Output Parsers", output_parsers)
            ]
            
            for name, module in modules:
                run_module(name, module)
                input("\nPress Enter to continue to next module...")
        
        else:
            print("\n⚠️  Invalid choice. Please enter a number between 0 and 8.")
        
        if choice != "0":
            input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)

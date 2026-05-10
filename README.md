# ChainItUp - LangChain Learning Application

A comprehensive, interactive application demonstrating all major features of LangChain. Perfect for beginners and intermediate developers looking to learn LangChain through practical, hands-on examples.

## 🚀 Features

This application covers all essential LangChain concepts:

- **Basic LLM Usage**: Initialize and use different LLM providers (OpenAI, Anthropic)
- **Prompt Templates**: Master dynamic prompt engineering with templates
- **Memory Management**: Handle conversation history with various memory types
- **Chains**: Build complex workflows using LangChain Expression Language (LCEL)
- **Agents & Tools**: Create autonomous AI systems with custom tools
- **RAG (Retrieval-Augmented Generation)**: Build document Q&A systems with vector stores
- **Output Parsers**: Structure and validate LLM outputs

## 📋 Prerequisites

- Python 3.9 or higher
- OpenAI API key (recommended)
- Anthropic API key (optional, for Claude examples)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ChainItUp
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API keys:
```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 🎯 Usage

### Web UI (Recommended)

Run the interactive web interface with Streamlit:
```bash
streamlit run app.py
```

The web UI provides:
- **Visual Interface**: Clean, modern interface with sidebar navigation
- **Interactive Demos**: Hands-on experience with LangChain features
- **Real-time Feedback**: See results instantly as you interact
- **Multiple Demos**: Basic LLM, Prompt Templates, Chains, Memory, and RAG

The web UI will open automatically in your browser at `http://localhost:8501`

### CLI Application

Run the command-line interactive application:
```bash
python main.py
```

You'll see an interactive menu where you can:
- Run individual modules to learn specific features
- Run all examples sequentially
- Exit at any time

### Running Individual Modules

You can also run individual example modules directly:

```bash
# Basic LLM usage
python examples/01_basic_llm.py

# Prompt templates
python examples/02_prompt_templates.py

# Memory management
python examples/03_memory.py

# Chains
python examples/04_chains.py

# Agents and tools
python examples/05_agents_tools.py

# RAG
python examples/06_rag.py

# Output parsers
python examples/07_output_parsers.py
```

## 📚 Module Descriptions

### 1. Basic LLM Usage (`01_basic_llm.py`)

Learn the fundamentals of using LLMs with LangChain:
- Simple calls to OpenAI and Anthropic models
- Using message history (system and human messages)
- Streaming responses
- Batch processing
- Temperature control for creativity

**Key Concepts**: LLM initialization, message types, streaming, batch processing

### 2. Prompt Templates (`02_prompt_templates.py`)

Master prompt engineering:
- Basic prompt templates with variables
- ChatPromptTemplate for structured conversations
- Partial templates with pre-filled values
- Message placeholders for conversation history
- Template composition
- Few-shot prompting

**Key Concepts**: PromptTemplate, ChatPromptTemplate, partial templates, few-shot learning

### 3. Memory Management (`03_memory.py`)

Handle conversation context:
- ConversationBufferMemory (stores all messages)
- ConversationBufferWindowMemory (keeps last k messages)
- ConversationSummaryMemory (summarizes conversation)
- ConversationKGMemory (extracts knowledge graph)
- Using memory with chains
- Comparing different memory types

**Key Concepts**: Memory types, conversation history, context management

### 4. Chains (`04_chains.py`)

Build complex workflows:
- Simple LLM chains using LCEL (|)
- Sequential chains (output of one → input of next)
- RunnablePassthrough for data flow
- RunnableLambda for custom transformations
- Branching chains with conditional logic
- Parallel chains
- Chains with memory
- Streaming through chains

**Key Concepts**: LCEL, RunnablePassthrough, RunnableLambda, sequential/parallel chains

### 5. Agents & Tools (`05_agents_tools.py`)

Create autonomous AI systems:
- Creating custom tools
- Using built-in tools (DuckDuckGo search)
- Tool-calling agents
- Agents with search capability
- Multi-tool agents
- Agents with memory
- Error handling in tools

**Key Concepts**: Tools, agents, AgentExecutor, tool-calling, error handling

### 6. RAG - Retrieval-Augmented Generation (`06_rag.py`)

Build document Q&A systems:
- Loading documents from various sources
- Text splitting strategies
- Creating embeddings
- Building vector stores
- Similarity search
- Complete RAG pipeline
- RAG with source citations
- Document QA workflow

**Key Concepts**: Document loaders, text splitters, embeddings, vector stores, retrievers

### 7. Output Parsers (`07_output_parsers.py`)

Structure LLM outputs:
- String output parser
- Pydantic output parser for structured data
- Comma-separated list parser
- JSON output parser
- Custom output parsers
- Error handling
- Multiple parsers in sequence
- Parser validation

**Key Concepts**: Output parsers, Pydantic models, structured output, validation

## 🏗️ Project Structure

```
ChainItUp/
├── app.py                  # Streamlit web UI (recommended)
├── main.py                 # CLI interactive application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .env                   # Your API keys (create this)
├── examples/
│   ├── __init__.py
│   ├── 01_basic_llm.py
│   ├── 02_prompt_templates.py
│   ├── 03_memory.py
│   ├── 04_chains.py
│   ├── 05_agents_tools.py
│   ├── 06_rag.py
│   └── 07_output_parsers.py
└── README.md              # This file
```

## 💡 Learning Path

We recommend following this order for optimal learning:

1. **Start with Basic LLM Usage** - Understand how to interact with LLMs
2. **Learn Prompt Templates** - Master prompt engineering
3. **Explore Memory** - Handle conversation context
4. **Build Chains** - Create complex workflows
5. **Use Agents & Tools** - Create autonomous systems
6. **Implement RAG** - Build document Q&A systems
7. **Master Output Parsers** - Structure LLM outputs

## 🔧 Configuration

### API Keys

You need at least one API key to run the examples:
- **OpenAI API Key**: Required for most examples (recommended)
- **Anthropic API Key**: Optional, for Claude-specific examples

Get your API keys:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

### Model Configuration

The examples use these default models:
- OpenAI: `gpt-4o-mini` (cost-effective, good for learning)
- Anthropic: `claude-3-5-sonnet-20241022`

You can modify the model in any example file to use different models.

## 🐛 Troubleshooting

### Common Issues

**"API key not found" error**
- Ensure you've created a `.env` file from `.env.example`
- Add your API key to the `.env` file
- Restart your terminal after creating the `.env` file

**"Module not found" error**
- Ensure you've installed all dependencies: `pip install -r requirements.txt`
- Make sure you're using the correct virtual environment

**ChromaDB persistence warning**
- This is normal for the examples which use in-memory vector stores
- The examples clean up after themselves automatically

**Rate limiting**
- If you hit rate limits, wait a few minutes before retrying
- Consider using a different model or reducing the number of examples

## 📖 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)

## 🤝 Contributing

This is a learning application. Feel free to:
- Add new examples
- Improve existing ones
- Fix bugs
- Enhance documentation

## 📝 License

This project is open source and available for educational purposes.

## 🎓 Tips for Learning

1. **Run examples sequentially** - Each module builds on previous concepts
2. **Read the code comments** - They explain what each part does
3. **Experiment** - Modify parameters and see what changes
4. **Check the output** - Understand what each example produces
5. **Build your own** - After learning, try creating your own chains/agents

## 🚀 Next Steps

After completing this tutorial, you'll be able to:
- Build chatbots with memory
- Create AI agents with tools
- Implement RAG systems for your documents
- Structure LLM outputs reliably
- Chain multiple components together
- Handle errors gracefully

Happy learning! 🎉

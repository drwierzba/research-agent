# LLM-Based Research Agent

## Description

LLM-Based Research Agent is an intelligent tool designed to automate and enhance the literature review process for researchers and academics. By leveraging advanced large language models (LLMs) and vector databases, this tool streamlines the discovery, analysis, and summarization of academic papers on any given research topic.

## Features

- **Semantic Paper Retrieval**: Automatically searches and retrieves relevant academic papers based on your research query.
- **Vector Database Storage**: Stores and indexes papers using embeddings for efficient similarity search.
- **Multimodal Document Analysis**: Analyzes PDF documents as images to maintain the integrity of figures, tables, and formatted text.
- **Comprehensive Summarization**: Generates insightful summaries highlighting key research themes, methodologies, findings, and potential areas for further research.
- **Customizable Focus**: Allows specifying additional focus areas for more targeted summaries.

## Installation

### Prerequisites
- Python 3.6+
- Virtual environment tool (virtualenv)

### Setup
1. Clone the repository:
```shell script
git clone https://github.com/your-username/research-agent.git
   cd research-agent
```


2. Create and activate a virtual environment:
```shell script
virtualenv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```


3. Install dependencies:
```shell script
pip install -r requirements.txt
```


## Usage

### Basic Usage
Run the agent with a research query:

```shell script
python main.py "Graph neural networks for traffic prediction"
```


### Advanced Options
You can customize your research with additional parameters:

```shell script
python main.py "Graph neural networks for traffic prediction" \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --paper-count 30 \
  --focus "Focus on comparative performance metrics and real-world applications" \
  --log-level INFO
```


### Command-line Arguments

- `query` (required): Research query or topic
- `--start-date`: Start date for paper search (YYYY-MM-DD)
- `--end-date`: End date for paper search (YYYY-MM-DD)
- `--paper-count`: Number of papers to retrieve (default: 20)
- `--focus`: Additional focus for the summary
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Architecture

The project follows a modular architecture:

- **ResearchAgent**: Main controller that orchestrates the research pipeline
- **PaperRetriever**: Handles paper retrieval from academic databases
- **ChromaVectorDb**: Manages vector embeddings and similarity search
- **MultimodalDocumentSummarizer**: Creates comprehensive summaries using multimodal LLMs
- **ModelAdapter**: Provides an interface to different LLM models

## How It Works

1. **Query Processing**: The system extracts keywords from your research query
2. **Paper Retrieval**: It searches academic databases for relevant papers
3. **Vector Embedding**: Papers are processed and stored in a vector database
4. **Similarity Search**: The most relevant papers are retrieved based on semantic similarity
5. **Multimodal Analysis**: PDFs are processed as images to maintain visual information
6. **Summary Generation**: A comprehensive summary is generated highlighting:
   - Main research themes and questions
   - Key methodologies
   - Main findings and conclusions
   - Contradictions or differences between papers
   - Potential areas for further research
   - Information about GitHub repositories related to the papers

## Example Output

```
Executing the research pipeline...
Query: Graph neural networks for traffic prediction
Searching articles in Semantic Scholar database (keywords: ['graph neural networks', 'traffic prediction', 'GNN'])...
Added embeddings for 10 new papers to the existing database

Querying vector database for similar papers...

Producing papers summary...

Summary:
The collected research papers focus on applying Graph Neural Networks (GNNs) to traffic prediction. Here's a synthesis of the main themes:

[Detailed summary would appear here with research themes, methodologies, findings, and potential future directions]
```


## Future Enhancements

- **Enhanced Paper Filtering**: Implement more advanced filtering mechanisms based on citation count, journal impact factor, or author reputation
- **Interactive Summaries**: Add interactive elements to summaries with clickable citations and expandable sections
- **Cross-Domain Connections**: Identify potential connections between papers from different domains
- **Research Gap Identification**: Automatically identify unexplored areas or gaps in the literature
- **Customizable Output Formats**: Support for different output formats like LaTeX, Markdown, or formatted PDF reports
- **Collaborative Research**: Enable multiple researchers to collaborate on the same literature review
- **Integration with Reference Managers**: Connect with reference management systems like Zotero or Mendeley

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
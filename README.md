RAG-based Q&A System for Lecture Transcripts

A Retrieval-Augmented Generation (RAG) system designed to convert lecture transcripts into a searchable knowledge base and answer questions intelligently. Perfect for students, educators, or anyone who wants quick answers from lecture content.

# Features

Download and parse lecture transcripts automatically.

Preprocess and clean text for indexing.

Build a knowledge base for fast retrieval.

Answer questions using Retrieval-Augmented Generation (RAG).

Easy environment setup with a single script.

# Repository Structure
File	Description
download_transcripts.py	Fetch lecture transcripts from sources.
parse_transcripts.py	Convert raw transcripts into structured data.
preprocess_passages.py	Clean and prepare text for indexing.
knowledge_base.py	Manage and store processed data.
retrieval_utils.py	Utilities for retrieving relevant information.
rag.py	Implements the Retrieval-Augmented Generation model for Q&A.
requirements.txt	List of Python dependencies.
setup.sh	Automates environment setup.
# Installation

Clone the repository

git clone https://github.com/shrav-n-9/RAG-based-Q-A-System.git
cd RAG-based-Q-A-System


Run setup script (installs dependencies automatically)

bash setup.sh


Or manually install dependencies

pip install -r requirements.txt

# Usage

Download transcripts

python download_transcripts.py


Parse transcripts

python parse_transcripts.py


Preprocess passages

python preprocess_passages.py


Build knowledge base

python knowledge_base.py


Ask questions using RAG

python rag.py

# How It Works

Transcript Processing
Lectures are fetched, parsed, and preprocessed into structured text passages.

Knowledge Base Creation
Passages are indexed for fast retrieval.

Retrieval-Augmented Generation
When a question is asked, the system retrieves relevant passages and uses a generative model to provide accurate answers.

# Customization

Add New Lectures: Drop transcripts into the data/ folder and rerun the pipeline.

Change RAG Model: Update the model path in rag.py to experiment with different generative models.

# Future Improvements

Web-based interface for live Q&A.

Support for multiple transcript formats (PDF, DOCX, etc.).

Enhanced evaluation and feedback system.

# Requirements

Python 3.10+

Libraries listed in requirements.txt

# Contribution

Feel free to open issues, submit PRs, or suggest features. Let's make learning from lectures smarter!
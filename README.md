# Gradio LLM Interface

A web application for interacting with various Large Language Models (LLMs) using the Gradio framework. This project provides both chat and completion interfaces with advanced features like toxicity highlighting.

## Features

- **Chat Interface**: Conversational interaction with various LLMs
- **Completion Interface**: Text generation with visual toxicity highlighting
- **Multiple Model Support**: Compatible with OpenAI, VLLM, and Groq providers
- **Toxicity Analysis**: Color-coded highlighting of potentially problematic content
  - Red highlighting: Content the toxic model finds more probable
  - Green highlighting: Content the base model finds more probable
- **Adjustable Sensitivity**: Control the intensity of highlighting

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gradio_site.git
cd gradio_site

# Install dependencies
pip install gradio openai
```

a
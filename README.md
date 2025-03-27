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
pip install -r requirements.txt

# Launch app
gradio main.py
```


Code used to launch vllm servers
```
VLLM_USE_V1=1 vllm serve enochlev/gemma-3-1b-pt-human --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization .2 --port 9300 --served-model-name gemma-3-1b-pt-human --no-enable-prefix-caching --max-num-batched-tokens 2048 --max-seq-len-to-capture 2048 --disable-log-stats



VLLM_USE_V1=1 vllm serve enochlev/gemma-3-1b-it-toxicity --max-model-len 1024 --max-num-seqs 1 --max_num_batched_tokens 1024 --gpu-memory-utilization .35 --port 9301 --served-model-name gemma-3-1b-it-toxicity --no-enable-prefix-caching --max-num-batched-tokens 1024 --max-seq-len-to-capture 1024 --disable-log-stats


VLLM_USE_V1=1 vllm serve google/gemma-3-1b-it --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization .5 --port 9302 --served-model-name gemma-3-1b-it --no-enable-prefix-caching --max-num-batched-tokens 2048 --max-seq-len-to-capture 2048 --disable-log-stats
```
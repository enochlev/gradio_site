import math
import os

import gradio as gr
import openai

    
base_url = os.environ.get("GROQ_BASE_URL")
api_key = os.environ.get("GROQ_API_KEY")

# New variable to control highlighting sensitivity
HUE_SCALE = .001  # Adjust this value to control highlight intensity

# Create OpenAI clients
# VLLM_USE_V1=1 vllm serve /home/enochlev/Documents/School/childs/gemma-3-1b-pt-human --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization .25 --port 9300 --swap-space 0 --served-model-name gemma-3-1b-pt-human
# VLLM_USE_V1=1 vllm serve /home/enochlev/Documents/School/childs/gemma-3-1b-it-toxicity --max-model-len 1024 --max-num-seqs 2 --gpu-memory-utilization .2 --port 9301 --swap-space 0 --served-model-name gemma-3-1b-it-toxicity
# VLLM_USE_V1=1 vllm serve google/gemma-3-1b-pt --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization .25 --port 9302 --swap-space 0 --served-model-name gemma-3-1b-pt
# VLLM_USE_V1=1 vllm serve google/gemma-3-1b-it --max-model-len 1024 --max-num-seqs 2 --gpu-memory-utilization .2 --port 9303 --swap-space 0 --served-model-name gemma-3-1b-it

clients = {}

clients["gemma-3-1b-pt-human"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9300", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-pt-human"
}

clients["gemma-3-1b-it-toxicity"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9301", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-it-toxicity"
}


clients["gemma-3-1b-pt"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9302", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-pt"
}

clients["gemma-3-1b-it"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9303", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-it"
}

space_token = "Ġ"

# Function for streaming chat responses
def chat_with_openai(message, history):
    # Add user message to history
    history = history + [(message, "")]
    
    # Prepare messages for API
    messages = []
    for user_msg, bot_msg in history:
        if user_msg:  # Skip empty messages
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:   # Skip empty messages (like the last one)
            messages.append({"role": "assistant", "content": bot_msg})
    
    # Stream the response
    try:
        stream = clients["gemma-3-1b-it"].chat.completions.create(
            model=clients["gemma-3-1b-it"]["chatbot_model"],
            messages=messages,
            stream=True,
            logprobs=1,
        )
        
        # Initialize response
        partial_response = ""
        
        # Stream the response token by token
        for chunk in stream:
            if chunk.choices[0].delta.content:
                partial_response += chunk.choices[0].delta.content
                # Update the last response in history
                history[-1] = (message, partial_response)
                yield history
    except Exception as e:
        history[-1] = (message, f"Error: {str(e)}")
        yield history

# Helper function to get color based on probability difference
def get_color_for_diff(diff):
    # diff > 0 means toxic model assigns higher probability (more toxic)
    # diff < 0 means base model assigns higher probability (more polite)
    
    # Scale the difference by HUE_SCALE to control sensitivity
    scaled_diff = min(max(-1, diff / HUE_SCALE), 1)
    
    if scaled_diff > 0:  # More toxic
        intensity = int(min(255, scaled_diff * 255))
        return f"rgba(255, {255-intensity}, {255-intensity}, 0.8)"
    else:  # More polite
        intensity = int(min(255, -scaled_diff * 255))
        return f"rgba({255-intensity}, 255, {255-intensity}, 0.8)"

# Modified completion function with toxicity highlighting
def completion_with_openai(text):
    try:  
        # Get logprobs for the prompt
        base_probs = clients["gemma-3-1b-pt"].completions.create(
            model=clients["gemma-3-1b-pt"]["chatbot_model"],
            prompt=text,
            logprobs=1,
            echo=True,
            max_tokens=0,
        ).choices[0].logprobs.token_logprobs


        tokens = base_probs.choices[0].logprobs.tokens
        tokens = [token.replace(space_token, " ") for token in tokens]
        base_logs = base_probs.choices[0].logprobs.token_logprobs

        # Stream the completion
        toxic_probs = clients["gemma-3-1b-it-toxicity"].completions.create(
            model=clients["gemma-3-1b-it-toxicity"]["chatbot_model"],
            prompt=text,
            logprobs=1,
            echo=True,
            max_tokens=0,
        ).choices[0].logprobs.token_logprobs
        

        all_tokens = toxic_probs.choices[0].logprobs.tokens
        formatted_response = ""
        for i in range(min_length):
            token = all_tokens[i]
            
            # Calculate difference between toxic and base model logprobs
            # Positive diff means toxic model gives higher probability (more toxic)
            if toxic_logs[i] == None or all_base_logs[i] == None:
                diff = 0
            else:
                diff = toxic_logs[i] - all_base_logs[i]
            
            # Get color based on the difference
            color = get_color_for_diff(diff)
            
            # Format the token with the appropriate color
            formatted_token = f"<span style='background-color: {color}'>{token}</span>"
            formatted_response += formatted_token
        
        # Return markdown with HTML styling
        yield f"<div style='font-family: monospace;'>{formatted_response}</div>"

    except Exception as e:
        yield text + f"\nError: {str(e)}"

# Create Gradio interface with tabs
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# OpenAI Assistant")
    
    with gr.Tabs():
        # Tab 1: Chat Interface
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=500)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type your message here...", lines=2,scale=5,show_label=False)
                submit = gr.Button("Send", scale=1,size="lg")
            clear = gr.Button("Clear")
            
            # Handle send message
            msg.submit(
                fn=chat_with_openai,
                inputs=[msg, chatbot],
                outputs=chatbot
            ).then(
                fn=lambda: "",
                outputs=msg
            )
            
            #if submit button is clicked
            submit.click(
                fn=chat_with_openai,
                inputs=[msg, chatbot],
                outputs=chatbot
            ).then(
                fn=lambda: "",
                outputs=msg
            )

            
            # Handle clear button
            clear.click(lambda: [], outputs=chatbot)
            
            gr.Examples(
                examples=["Tell me about artificial intelligence", "Write a short poem about technology"],
                inputs=msg
            )
        
        # Completion tab with HTML-highlighted text
        with gr.Tab("Completion"):
            completion_text = gr.HTML(label="AI Completion with Toxicity Highlighting")
            input_text = gr.Textbox(placeholder="Enter your prompt here...", lines=5, label="Prompt")
            
            with gr.Row():
                generate = gr.Button("Generate")
                clear_completion = gr.Button("Clear")
            
            # Slider to adjust highlighting sensitivity
            hue_scale_slider = gr.Slider(
                minimum=.1, 
                maximum=2.0, 
                value=.1, 
                step=0.1, 
                label="Highlighting Sensitivity (HUE_SCALE)"
            )
            
            # Update HUE_SCALE when slider changes
            def update_hue_scale(value):
                global HUE_SCALE
                HUE_SCALE = value
                return f"HUE_SCALE set to: {value}"
                
            hue_scale_slider.change(
                fn=update_hue_scale,
                inputs=hue_scale_slider,
                outputs=gr.Textbox(label="Status")
            )
            
            # Generate with streaming
            generate.click(
                fn=completion_with_openai,
                inputs=input_text,
                outputs=completion_text,
                api_name="generate_completion",
                queue=True
            )
            
            # Handle clear button
            clear_completion.click(lambda: "", outputs=input_text)
            clear_completion.click(lambda: "", outputs=completion_text)

# Launch the demo
demo.launch(debug=True)

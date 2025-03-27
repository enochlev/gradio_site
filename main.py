import math
import os

import gradio as gr
import openai
import re

    
base_url = os.environ.get("GROQ_BASE_URL")
api_key = os.environ.get("GROQ_API_KEY")

# New variable to control highlighting sensitivity
HUE_SCALE = 1.0 

# Create OpenAI clients
# VLLM_USE_V1=1 vllm serve /home/enochlev/Documents/School/childs/gemma-3-1b-pt-human --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization .2 --port 9300 --served-model-name gemma-3-1b-pt-human --no-enable-prefix-caching --max-num-batched-tokens 2048 --max-seq-len-to-capture 2048 --disable-log-stats
# VLLM_USE_V1=1 vllm serve /home/enochlev/Documents/School/childs/gemma-3-1b-it-toxicity --max-model-len 1024 --max-num-seqs 1 --max_num_batched_tokens 1024 --gpu-memory-utilization .35 --port 9301 --served-model-name gemma-3-1b-it-toxicity --no-enable-prefix-caching --max-num-batched-tokens 1024 --max-seq-len-to-capture 1024 --disable-log-stats
# VLLM_USE_V1=1 vllm serve google/gemma-3-1b-pt --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization .5 --port 9302 --served-model-name gemma-3-1b-pt --no-enable-prefix-caching --max-num-batched-tokens 2048 --max-seq-len-to-capture 2048 --disable-log-stats
# VLLM_USE_V1=1 vllm serve google/gemma-3-1b-it --max-model-len 1024 --max-num-seqs 1 --max_num_batched_tokens 1024 --gpu-memory-utilization .65 --port 9303 --served-model-name gemma-3-1b-it --no-enable-prefix-caching --max-num-batched-tokens 1024 --max-seq-len-to-capture 1024 --disable-log-stats

clients = {}

clients["gemma-3-1b-pt-human"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9300/v1", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-pt-human"
}

clients["gemma-3-1b-it-toxicity"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9301/v1", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-it-toxicity"
}


clients["gemma-3-1b-pt"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9302/v1", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-pt"
}

clients["gemma-3-1b-it"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9303/v1", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-it"
}

space_token = "▁"
start_of_turn_user = "<start_of_turn>user\n"
start_of_turn_model = "<start_of_turn>model\n"
end_of_turn = "<end_of_turn>"

# Function for streaming chat responses
# Function for streaming chat responses with toxicity highlighting
def chat_with_openai(message, history):
    # Add user message to history
    history = history + [(message, "")]
    
    # Prepare messages for API
    messages = []
    for user_msg, bot_msg in history:
        if user_msg:  # Skip empty messages
            #remove all color tags
            user_msg = re.sub(r'''<span style=("|')background-color: rgba\(\d+, \d+, \d+, (\d|\.|e|-)+\)("|')>|<\/span>''', '', user_msg)

            messages.append({"role": "user", "content": user_msg})
        if bot_msg:  # Skip empty messages and strip HTML
            bot_msg = re.sub(r'''<span style=("|')background-color: rgba\(\d+, \d+, \d+, (\d|\.|e|-)+\)("|')>|<\/span>''', '', bot_msg)
            messages.append({"role": "assistant", "content": bot_msg})
    
    # Stream the response
    stream = clients["gemma-3-1b-it"]["client"].chat.completions.create(
        model=clients["gemma-3-1b-it"]["chatbot_model"],
        messages=messages,
        stream=True,
        logprobs=1,
    )
    
    # Initialize response
    partial_response = ""
    
    # Stream the response token by token first
    for chunk in stream:
        if chunk.choices[0].delta.content:
            partial_response += chunk.choices[0].delta.content
            # Show unformatted response while streaming
            history[-1] = (message, partial_response)
            yield history
    messages.append({"role": "assistant", "content": partial_response})
    # After streaming is complete, apply highlighting
    # Get logprobs for the complete response from both models
    base_result = clients["gemma-3-1b-it"]["client"].chat.completions.create(
        model=clients["gemma-3-1b-it"]["chatbot_model"],
        messages=messages,
        max_tokens=1,
        extra_body={"prompt_logprobs": True},
    )
    
    
    toxic_result = clients["gemma-3-1b-it-toxicity"]["client"].chat.completions.create(
        model=clients["gemma-3-1b-it-toxicity"]["chatbot_model"],
        messages=messages,
        max_tokens=1,
        extra_body={"prompt_logprobs": True},
    )

    
    # Process tokens and apply highlighting
    base_result = [list(i.values())[0] for i in base_result.model_extra['prompt_logprobs'] if i is not None]
    base_probs = [i['logprob'] for i in base_result if i is not None]
    base_tokens = [i['decoded_token'].replace(space_token, " ") for i in base_result if i is not None]

    toxic_result = [list(i.values())[0] for i in toxic_result.model_extra['prompt_logprobs'] if i is not None]
    toxic_probs = [i['logprob'] for i in toxic_result if i is not None]
    toxic_tokens = [i['decoded_token'].replace(space_token, " ") for i in toxic_result if i is not None]
    
    formatted_response = ""
    formatted_chat = []
    turn = None # None / user / assisstant
    for i in range(len(base_tokens)):
        extra_formatted_response = formatted_response + base_tokens[i]
        if turn is None:
            if extra_formatted_response.endswith(start_of_turn_user):
                turn = "user"
                formatted_chat.append(["",None])
            elif extra_formatted_response.endswith(start_of_turn_model):
                turn = "assistant"
                formatted_chat.append([None,""])

            formatted_response += base_tokens[i]
            continue

        if turn == "user":
            if extra_formatted_response.endswith(end_of_turn):
                turn = None
                formatted_response += base_tokens[i]
                continue
        elif turn == "assistant":
            if extra_formatted_response.endswith(end_of_turn):
                turn = None
                formatted_response += base_tokens[i]
                continue
            
        
        
        # Calculate difference between toxic and base model logprobs
        diff = toxic_probs[i] - base_probs[i]
        color = get_color_for_diff(diff)

        token = base_tokens[i]

        if turn == "user":
            current_chat = formatted_chat[-1][0]
        else:
            current_chat = formatted_chat[-1][1]

        

        
        # Format token with appropriate color
        if "\n" in token:
            # Handle newlines by replacing them with <br> tags
            current_chat += token
            if False:
                parts = token.split("\n")
                for i, part in enumerate(parts):
                    if part:  # Only add span if there's content
                        current_chat += f"<span style='background-color: {color}'>{part}</span>"
                    # Add line break after each part except the last one
                    if i < len(parts) - 1:
                        current_chat += "<br>"
        else:
            current_chat += f"<span style='background-color: {color}'>{token}</span>"
    
        # Update chat history with highlighted response
        if turn == "user":
            formatted_chat[-1][0] = current_chat
        else:
            formatted_chat[-1][1] = current_chat
    
        yield formatted_chat



    # for i in range(formatted_chat):
    #     #history[-1] = (message, f"<div style='line-height: 1.5;'>{formatted_response + "".join(tokens[i+1:])}</div>")
    #     if current_chat[-1][0] is not None:
    #         current_chat[-1][0] = f"<div style='line-height: 1.5;'>{current_chat[-1][0]}</div>"
    #     if current_chat[-1][1] is not None:
    #         current_chat[-1][1] = f"<div style='line-height: 1.5;'>{current_chat[-1][1]}</div>"
    
    # yield current_chat
    

# Helper function to get color based on probability difference
def get_color_for_diff(diff):
    # diff > 0 means toxic model assigns higher probability (more toxic)
    # diff < 0 means base model assigns higher probability (more polite)
    
    # Scale the difference by HUE_SCALE to control sensitivity
    scaled_diff = min(max(-1, diff / HUE_SCALE), 1)
    if scaled_diff > 0:  # more lora
        # Use red with transparency based on intensity
        transparency = min(0.9, abs(scaled_diff) * 0.9)  # Cap at 0.9 for some visibility
        return f"rgba(150, 0, 0, {transparency})"
    else:  # less lora
        # Use green with transparency based on intensity
        transparency = min(0.9, abs(scaled_diff) * 0.9)  # Cap at 0.9 for some visibility
        return f"rgba(0, 150, 0, {transparency})"

# Modified completion function with toxicity highlighting
def completion_with_openai(text):
    try:  
        # Get logprobs for the prompt
        base = clients["gemma-3-1b-pt"]["client"].completions.create(
            model=clients["gemma-3-1b-pt"]["chatbot_model"],
            prompt=text,
            logprobs=1,
            echo=True,
            max_tokens=0,
        )

        tokens = base.choices[0].logprobs.tokens
        tokens = [token.replace(space_token, " ") for token in tokens]
        base_logs = base.choices[0].logprobs.token_logprobs

        # Stream the completion
        lora_probs = clients["gemma-3-1b-it-toxicity"]["client"].completions.create(
            model=clients["gemma-3-1b-it-toxicity"]["chatbot_model"],
            prompt=text,
            logprobs=1,
            echo=True,
            max_tokens=0,
        ).choices[0].logprobs.token_logprobs
        
        formatted_response = ""
        for i in range(len(tokens)):
            if tokens[i] in ["<bos>", "<eos>"]:
                continue

            token = tokens[i]
            
            # Calculate difference between toxic and base model logprobs
            if lora_probs[i] == None or base_logs[i] == None:
                diff = 0
            else:
                diff = lora_probs[i] - base_logs[i]
            
            # Get color based on the difference
            color = get_color_for_diff(diff)
            
            # Format the token with the appropriate color
            if color is None:
                formatted_response += token
            elif "\n" in token:
                # Handle newlines by replacing them with <br> tags
                formatted_response += token
                if False:
                    parts = token.split("\n")
                    for j, part in enumerate(parts):
                        if part:  # Only add span if there's content
                            formatted_response += f"<span style='background-color: {color}'>{part}</span>"
                        # Add line break after each part except the last one
                        if j < len(parts) - 1:
                            formatted_response += "<br>"
            else:
                formatted_response += f"<span style='background-color: {color}'>{token}</span>"
        
        # Return with improved styling
        #yield f"<div style='font-family: \"Inter\", \"Segoe UI\", Arial, sans-serif; line-height: 1.5;'>{formatted_response}</div>"
        yield formatted_response
    except Exception as e:
        yield text + f"\nError: {str(e)}"

# Create Gradio interface with tabs
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# AI Playground")
    
    with gr.Tabs():
        # Tab 1: Chat Interface
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=500)#, #type="messages")
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
            #label title: "ChatGPT written essay detector"
            label = gr.Markdown("# OpenAI Completion with Bot Detection Highlighting")
            completion_text = gr.HTML(label="AI Completion with Bot Detection  Highlighting")
            input_text = gr.Textbox(placeholder="Enter your prompt here...", lines=5, label="Prompt")
            
            with gr.Row():
                generate = gr.Button("Generate")
                clear_completion = gr.Button("Clear")
            
            # Slider to adjust highlighting sensitivity
            hue_scale_slider = gr.Slider(
                minimum=.1, 
                maximum=2.0, 
                value=HUE_SCALE, 
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

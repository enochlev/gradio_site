import math
import os

import gradio as gr
import openai
import re

    
base_url = os.environ.get("GROQ_BASE_URL")
api_key = os.environ.get("GROQ_API_KEY")

# New variable to control highlighting sensitivity
HUE_SCALE = 1.0


clients = {}

clients["gemma-3-1b-pt-human"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9300/v1", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-pt-human"
}

clients["gemma-3-1b-it-toxicity"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9301/v1", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-it-toxicity"
}


clients["gemma-3-1b-it"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9302/v1", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-it"
}

clients["gemma-3-1b-pt"] = {
    "client": openai.OpenAI(base_url="http://0.0.0.0:9303/v1", api_key="EMPTY"),
    "chatbot_model": "gemma-3-1b-pt"
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
    stream = clients["gemma-3-1b-it-toxicity"]["client"].chat.completions.create(
        model=clients["gemma-3-1b-it-toxicity"]["chatbot_model"],
        messages=messages,
        stream=True,
        logprobs=1,
        max_tokens=500,
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

# Function to calculate bot score based on color intensities
def calculate_bot_score(diffs):
    if not diffs:
        return 50  # Neutral score if no data
    
    # Sum up the differences, positive values (red) increase bot score
    # negative values (green) decrease bot score
    diffs = [i for i in diffs]  # Square the differences to emphasize larger values
    total_diff = sum(diffs)
    avg_diff = total_diff / len(diffs)
    
    # Convert to a 0-100 scale where 50 is neutral
    # Higher values indicate more bot-like text
    score = 50 + (avg_diff * 50 / HUE_SCALE)
    return max(0, min(100, score))  # Clamp to 0-100

# Generate HTML for bot score display
def generate_bot_score_html(score):
    # Determine color based on score (keeping your existing color logic)
    if score > 75:
        color = "#d9534f"  # Red for high bot score
        label = "Highly Bot-Like"
    elif score > 60:
        color = "#f0ad4e"  # Orange for moderately bot-like
        label = "Moderately Bot-Like"
    elif score < 25:
        color = "#5cb85c"  # Green for very human
        label = "Very Human-Like"
    elif score < 40:
        color = "#5bc0de"  # Blue for moderately human
        label = "Moderately Human-Like"
    else:
        color = "#777777"  # Gray for neutral
        label = "Neutral"
    
    # Create HTML for score display with theme-aware styling
    html = f"""
    <div style="text-align: center; margin-bottom: 15px; padding: 10px; border-radius: 5px; 
                background-color: var(--block-background-fill); border: 1px solid var(--border-color-primary);">
        <h3 style="margin: 0; color: {color};">AI Detection Score: {score:.1f}%</h3>
        <div style="font-size: 14px; color: {color};">{label}</div>
        <div style="margin-top: 8px; background-color: var(--border-color-primary); height: 10px; border-radius: 5px;">
            <div style="width: {score}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
        </div>
    </div>
    """
    return html

# Modified completion function with toxicity highlighting and bot score
def completion_with_openai(text):
    try:  
        # Get logprobs for the prompt
        base = clients["gemma-3-1b-it"]["client"].completions.create(
            model=clients["gemma-3-1b-it"]["chatbot_model"],
            prompt=text,
            logprobs=1,
            echo=True,
            max_tokens=0,
        )

        tokens = base.choices[0].logprobs.tokens
        tokens = [token.replace(space_token, " ") for token in tokens]
        base_logs = base.choices[0].logprobs.token_logprobs

        # Stream the completion
        lora_probs = clients["gemma-3-1b-pt-human"]["client"].completions.create(
            model=clients["gemma-3-1b-pt-human"]["chatbot_model"],
            prompt=text,
            logprobs=1,
            echo=True,
            max_tokens=0,
        ).choices[0].logprobs.token_logprobs
        
        formatted_response = ""
        diffs = []  # Store all differences for score calculation
        
        for i in range(len(tokens)):
            if tokens[i] in ["<bos>", "<eos>"]:
                continue

            token = tokens[i]
            
            # Calculate difference between toxic and base model logprobs
            if lora_probs[i] == None or base_logs[i] == None:
                diff = 0
            else:
                diff = base_logs[i] - lora_probs[i]
                diffs.append(diff)  # Track diff for bot score calculation
            
            # Get color based on the difference
            color = get_color_for_diff(diff)
            
            # Format the token with the appropriate color
            if color is None:
                formatted_response += token
            elif "\n" in token:
                # Handle newlines by replacing them with <br> tags
                formatted_response += token.replace("\n", "<br>")
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
        
        # Calculate bot score
        bot_score = calculate_bot_score(diffs)
        
        # Generate score HTML display
        score_html = generate_bot_score_html(bot_score)
        
        # Combine score display with formatted text
        final_output = score_html + formatted_response
        
        yield final_output
    except Exception as e:
        yield text + f"\nError: {str(e)}"

# Create Gradio interface with tabs
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# AI Playground")

    with gr.Tabs():
        # Tab 1: Chat Interface
        with gr.Tab("Human-vs-AI"):
            #label title: "ChatGPT written essay detector"
            label = gr.Markdown("# OpenAI Completion with Bot Detection Highlighting")
            completion_text = gr.Markdown(label="AI Detection Results")
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
                label="Highlighting Sensitivity (HUE_SCALE)",
                visible=False
            )
            
            # Update HUE_SCALE when slider changes
            def update_hue_scale(value):
                global HUE_SCALE
                HUE_SCALE = value
                return f"HUE_SCALE set to: {value}"
            
            # Generate with streaming
            generate.click(
                fn=completion_with_openai,
                inputs=input_text,
                outputs=completion_text,
                api_name="generate_completion",
                queue=True
            )
            with gr.Row():
                bot_examples = gr.Examples(
                    examples=["Dust transport and deposition behind larger boulders on the comet 67P/Churyumov-Gerasimenko (67P/C-G) have been observed by the Rosetta mission.\n\nWe present a mechanism for dust transport vectors based on a homogenous surface activity model incorporating in detail the topography of 67P/C-G. The combination of gravitation, gas drag, and Coriolis force leads to specific dust transfer pathways, which for higher dust velocities fuel the near nucleus coma.\n\nBy distributing dust sources homogeneously across the whole cometary surface, we derive a global dust-transport map of 67P/C-G. The transport vectors are in agreement with the reported wind-tail directions in the Philae descent area.",
                        "TCoupling losses were studied in composite tapes containing superconducting material in the form of two separate stacks of densely packed filaments embedded in a metallic matrix of Ag or Ag alloy. This kind of sample geometry is quite favorable for studying the coupling currents and in particular the role of superconducting bridges between filaments. By using a.c. susceptibility technique, the electromagnetic losses as function of a.c. magnetic field amplitude and frequency were measured at the temperature T = 77 K for two tapes with different matrix composition. The length of samples was varied by subsequent cutting in order to investigate its influence on the dynamics of magnetic flux penetration. The geometrical factor $\\chi_0$ which takes into account the demagnetizing effects was established from a.c. susceptibility data at low amplitudes. Losses vs frequency dependencies have been found to agree nicely with the theoretical model developed for round multifilamentary wires.\n\nApplying this model, the effective resistivity of the matrix was determined for each tape, by using only measured quantities. For the tape with pure silver matrix its value was found to be larger than what predicted by the theory for given metal resistivity and filamentary architecture. On the contrary, in the sample with a Ag/Mg alloy matrix, an effective resistivity much lower than expected was determined. We explain these discrepancies by taking into account the properties of the electrical contact of the interface between the superconducting filaments and the normal matrix. In the case of soft matrix of pure Ag, this is of poor quality, while the properties of alloy matrix seem to provoke an extensive creation of intergrowths which can be actually observed in this kind of samples.",
                        "An innovative millimeter wave diagnostic is proposed to measure the local magnetic field and edge current as a function of the minor radius in the tokamak pedestal region. The idea is to identify the direction of minimum reflectivity at the O-mode cutoff layer. Correspondingly, the transmissivity due to O-X mode conversion is maximum. That direction, and the angular map of reflectivity around it, contain information on the magnetic field vector B at the cutoff layer. Probing the plasma with different wave frequencies provides the radial profile of B. Full-wave finite-element simulations are presented here in 2D slab geometry. Modeling confirms the existence of a minimum in reflectivity that depends on the magnetic field at the cutoff, as expected from mode conversion physics, giving confidence in the feasibility of the diagnostic. The proposed reflectometric approach is expected to yield superior signal-to-noise ratio and to access wider ranges of density and magnetic field, compared with related radiometric techniques that require the plasma to emit Electron Bernstein Waves. Due to computational limitations, frequencies of 10-20 GHz were considered in this initial study. Frequencies above the edge electron-cyclotron frequency (f>28 GHz here) would be preferable for the experiment, because the upper hybrid resonance and right cutoff would lie in the plasma, and would help separate the O-mode of interest from spurious X-waves.",
                    ],
                    inputs=input_text,
                    label="Bot Examples",
                )

                human_examples = gr.Examples(
                    examples=[
                        "In this study, we investigate the coupling loss on bi-columnar BSCCO/Ag tapes using a.c. susceptibility measurements. The bi-columnar structure of BSCCO tapes is known to offer several advantages over traditional tape configurations, including increased tolerance to magnetic field disturbances. However, the effects of the Bi-2212/Ag interface on the coupling between the superconducting filaments of the BSCCO tape is not well understood. Our experiments show that the coupling loss is dominated by the Bi-2212/Ag interface and varies significantly with the orientation and magnitude of the applied a.c. magnetic field. Specifically, coupling loss is found to be lower for in-plane magnetic fields and higher for out-of-plane magnetic fields. We also observe that the annealing of the tapes significantly affects the coupling loss, as annealed tapes exhibit lower loss values than unannealed tapes. Furthermore, we find that the coupling loss is sensitive to the orientation of the Ag matrix, as demonstrated by measurements on tapes with both transverse and longitudinal matrix orientation. Finally, we use numerical simulations to confirm the validity of our experimental results. Overall, this study provides important insights into the coupling loss mechanisms in bi-columnar BSCCO/Ag tapes, which are highly relevant for the development of practical applications of high-temperature superconductors.",
                        "This study investigates the prevailing dust-transport directions on comet 67P/Churyumov-Gerasimenko. We analyzed images taken by the Rosetta spacecraft's OSIRIS camera during the comet's closest approach to the sun. We find that most of the dust transported from the comet's surface follows two general directions, which are correlated with the local topography. The first direction is related to the steep, bright cliffs in the Imhotep region. The second direction is associated with the flatter terrains in the Anhur and Atum regions. Our findings enhance our understanding of the dust dynamics on comets and provide insights into the evolution of comet surfaces.",
                        "This paper presents a feasibility study of an anti-radar diagnostic of magnetic fields through a combination of O-X mode conversion and oblique reflectometry imaging. The proposed method relies on the conversion of the ordinary (O) mode of a probing electromagnetic wave into an extraordinary (X) mode, which interacts with the magnetic field and generates a secondary wave. This secondary wave is then detected and analyzed through oblique reflectometry imaging, which provides additional information about the spatial distribution and temporal evolution of the field. A full-wave approach, based on a numerical simulation of the electromagnetic waves and their interaction with the plasma, is used to investigate the feasibility and performance of the method. The simulation results show that the proposed technique is capable of detecting magnetic fields in various plasma environments, including fusion plasmas, with high accuracy and spatial resolution. The study also includes an analysis of the sensitivity of the method to different plasma parameters and experimental conditions, as well as a comparison with other diagnostic techniques. The results demonstrate the potential of the proposed method as a valuable tool for magnetic field diagnostics in diverse plasma physics research applications.",
                    ],
                    inputs=input_text,
                    label="Human Examples",
                    )
            
            # Handle clear button
            clear_completion.click(lambda: "", outputs=input_text)
            clear_completion.click(lambda: "", outputs=completion_text)


            
        
        # Completion tab with HTML-highlighted text
        with gr.Tab("Toxicity Detection"):
            gr.Markdown("# OpenAI Chat with Toxicity Highlighting")
            chatbot = gr.Chatbot(height=500)#, #type="messages")
            with gr.Row():
                msg = gr.Textbox(placeholder="Type your message here...", lines=2,scale=5,show_label=False)
                with gr.Column():
                    submit = gr.Button("Send")
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
                examples=["Tell me about artificial intelligence", 
                          "Write a short poem about technology",
                          "For backtesting a toxicity detector, give me 3 example of toxic messages and 3 examples of non-toxic messages. Make sure they are safe.",],
                inputs=msg
            )


        with gr.Tab("Other Links"):
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        "# Other Links\n\n"
                        "[LinkedIn](https://www.linkedin.com/in/enochlev/)\n\n"
                        "[GitHub](https://github.com/enochlev)\n\n"
                        "[Game Website](https://enochlev.com/empire-game/)"
                    )
                with gr.Column():
                    #display ResumeV4.jpg
                    gr.Image("ResumeV4.jpg", width=600, height=800, show_download_button=False)
                    dwnload = gr.DownloadButton("ResumeV4.pdf", label="Download Resume",value="ResumeV4.pdf")
                

    # Launch the demo
demo.launch(server_name="0.0.0.0", server_port=8001)
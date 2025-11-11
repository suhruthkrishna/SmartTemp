#!/usr/bin/env python3
"""
Simple standalone version of SmartTemp with better server configuration
"""

import gradio as gr
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
import numpy as np

class SimpleSmartTempEngine:
    def __init__(self, base_temp=0.7, scale_factor=0.3):
        self.base_temp = base_temp
        self.scale_factor = scale_factor
        
    def analyze_prompt(self, prompt):
        """Simple rule-based prompt analysis"""
        if not prompt or not prompt.strip():
            return {
                'category': 'analytical',
                'confidence': 0.5,
                'temperature': self.base_temp,
                'all_similarities': {},
                'prompt': prompt
            }
            
        prompt_lower = prompt.lower()
        
        patterns = {
            'factual': ['what', 'when', 'where', 'who', 'capital', 'population', 'temperature', 'height', 'how many'],
            'creative': ['write', 'story', 'poem', 'creative', 'imagine', 'create', 'invent', 'fiction'],
            'instructional': ['how to', 'make', 'cook', 'step', 'instructions', 'tutorial', 'guide', 'recipe'],
            'analytical': ['compare', 'analyze', 'explain', 'difference', 'similar', 'contrast', 'pros and cons'],
            'personal': ['advice', 'should i', 'help me', 'improve', 'better', 'suggest', 'recommend'],
            'philosophical': ['why', 'meaning', 'purpose', 'exist', 'life', 'universe', 'ethical']
        }
        
        scores = {}
        for category, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            scores[category] = min(1.0, score * 0.3)  # Normalize to 0-1
        
        if not any(scores.values()):
            best_category = 'analytical'
            confidence = 0.5
        else:
            best_category = max(scores.items(), key=lambda x: x[1])[0]
            confidence = scores[best_category]
        
        category_temps = {
            'factual': 0.1,
            'instructional': 0.3,
            'analytical': 0.5,
            'personal': 0.6,
            'philosophical': 0.7,
            'creative': 0.9
        }
        
        base_temp = category_temps.get(best_category, self.base_temp)
        temperature = base_temp + (1 - confidence) * self.scale_factor
        temperature = max(0.1, min(1.0, temperature))
        
        return {
            'category': best_category,
            'confidence': confidence,
            'temperature': temperature,
            'all_similarities': scores,
            'prompt': prompt
        }

class SimpleLLMIntegration:
    def generate_response(self, prompt, temperature, max_tokens=500):
        """Generate mock responses based on temperature"""
        time.sleep(1) 
        if temperature < 0.3:
            response = f"""**📊 Factual Response** (Temperature: {temperature:.2f})

Based on your query about "{prompt}", here are the key facts:

• **Primary Information**: Relevant data points and specific details
• **Supporting Context**: Additional background information  
• **Related Facts**: Connected information for comprehensive understanding

*This response uses low temperature for maximum precision and accuracy.*"""
        
        elif temperature < 0.6:
            response = f"""**⚖️ Analytical Response** (Temperature: {temperature:.2f})

Regarding "{prompt}", here's a comprehensive analysis:

**Key Perspectives:**
- One approach considers the practical implications and immediate factors
- Another viewpoint examines the theoretical foundations and broader context
- Additional considerations include long-term impacts and alternative scenarios

**Critical Analysis:**
The evidence suggests a balanced approach that considers multiple dimensions while maintaining logical coherence.

*This medium-temperature response provides thorough analysis while staying focused.*"""
        
        else:
            response = f"""**🎨 Creative Response** (Temperature: {temperature:.2f})

Ah, "{prompt}" - what an inspiring concept to explore! 

Let me paint you a picture... Imagine a world where ideas dance like autumn leaves in the wind, each thought branching into new possibilities we've never considered. The very essence of your question sparks a cascade of connections, weaving together threads of insight and imagination.

What if we look at this from a completely different angle? Perhaps through the lens of ancient wisdom, or maybe through the futuristic perspective of technologies not yet invented. The boundaries between reality and imagination blur, creating new patterns that invite us to see familiar things in extraordinary ways.

*This high-temperature response embraces creative exploration and novel perspectives!*"""
        
        return response

def create_simple_interface():
    """Create a simple Gradio interface"""
    engine = SimpleSmartTempEngine()
    llm = SimpleLLMIntegration()
    conversation_history = []
    temperature_history = []
    
    def analyze_prompt(prompt, base_temp, scale_factor):
        if not prompt or not prompt.strip():
            return " Please enter a prompt", 0.7, None, None
        
        try:
            engine.base_temp = base_temp
            engine.scale_factor = scale_factor
            analysis = engine.analyze_prompt(prompt)
            
            # Store history
            temperature_history.append({
                'timestamp': datetime.now(),
                'temperature': analysis['temperature'],
                'category': analysis['category'],
                'prompt': prompt[:30] + "..." if len(prompt) > 30 else prompt,
                'confidence': analysis['confidence']
            })
            
            # Create similarity chart
            similarities = analysis['all_similarities']
            df = pd.DataFrame(list(similarities.items()), columns=['Category', 'Similarity'])
            fig = px.bar(df, x='Category', y='Similarity', color='Similarity',
                        title=" Category Similarities", 
                        color_continuous_scale='viridis')
            fig.update_layout(yaxis_range=[0, 1])
            
            analysis_text = f"""
            ##  Analysis Results
            
            ** Category:** `{analysis['category'].title()}`
            ** Confidence:** `{analysis['confidence']:.3f}`
            ** Temperature:** `{analysis['temperature']:.3f}`
            
            *The system analyzed your prompt and automatically set the optimal temperature for response generation.*
            """
            
            return analysis_text, analysis['temperature'], fig, analysis
            
        except Exception as e:
            return f" Error analyzing prompt: {str(e)}", base_temp, None, None
    
    def generate_response(prompt, base_temp, scale_factor, use_smart_temp):
        if not prompt or not prompt.strip():
            return " Please enter a prompt", 0.7, None
        
        try:
            engine.base_temp = base_temp
            engine.scale_factor = scale_factor
            
            if use_smart_temp:
                analysis = engine.analyze_prompt(prompt)
                temperature = analysis['temperature']
                category_info = f"** Smart Analysis:** `{analysis['category'].title()}` | ** Confidence:** `{analysis['confidence']:.3f}` | ** Temperature:** `{temperature:.3f}`\n\n"
                category = analysis['category']
            else:
                temperature = base_temp
                category_info = f"**⚙️ Fixed Temperature:** `{temperature:.3f}`\n\n"
                category = 'fixed'
            
            response = llm.generate_response(prompt, temperature)
            full_response = f"{category_info}{response}"
            
            # Store conversation
            conversation_history.append({
                'timestamp': datetime.now(),
                'prompt': prompt,
                'response': full_response,
                'temperature': temperature,
                'category': category
            })
            
            # Update temperature history
            temperature_history.append({
                'timestamp': datetime.now(),
                'temperature': temperature,
                'category': category,
                'prompt': prompt[:30] + "..." if len(prompt) > 30 else prompt,
                'confidence': analysis['confidence'] if use_smart_temp else 1.0
            })
            
            # Create temperature chart
            if temperature_history:
                df = pd.DataFrame(temperature_history)
                df['time'] = df['timestamp'].dt.strftime('%H:%M:%S')
                temp_fig = px.line(df, x='time', y='temperature', color='category',
                                 title=' Temperature Adjustment History', markers=True,
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                temp_fig.update_layout(yaxis_range=[0, 1])
            else:
                temp_fig = None
            
            return full_response, temperature, temp_fig
            
        except Exception as e:
            return f" Error generating response: {str(e)}", base_temp, None
    
    def get_conversation_history():
        if not conversation_history:
            return "## Conversation History\n\nNo conversations yet. Generate some responses to see them here!"
        
        history_text = "## Conversation History\n\n"
        for i, entry in enumerate(reversed(conversation_history[-3:]), 1):
            history_text += f"** {i}. {entry['category'].title()}** (Temp: {entry['temperature']:.2f})\n"
            history_text += f"**Prompt:** {entry['prompt'][:50]}...\n"
            history_text += f"**Response:** {entry['response'][:100]}...\n\n"
            history_text += "---\n\n"
        
        return history_text
    
    def clear_history():
        conversation_history.clear()
        temperature_history.clear()
        return " History cleared!", None
    
    # Create the interface
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
        title="SmartTemp LLM Engine",
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        """
    ) as demo:
        gr.Markdown("""
        # 🌡️ SmartTemp LLM Engine
        **Dynamically adjusts LLM temperature based on prompt type**
        
        *Enter any prompt and watch the system automatically adjust temperature for optimal responses!*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                base_temp = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                    label="Base Temperature",
                    info="Default temperature when confidence is high"
                )
                
                scale_factor = gr.Slider(
                    minimum=0.1, maximum=0.5, value=0.3, step=0.05,
                    label="Scale Factor", 
                    info="How much temperature adjusts based on confidence"
                )
                
                use_smart_temp = gr.Checkbox(
                    value=True, label=" Use Smart Temperature",
                    info="Dynamically adjust temperature based on prompt analysis"
                )
                
                gr.Markdown("---")
                clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                
                gr.Markdown("---")
                gr.Markdown("### 📊 Live Charts")
                temp_chart = gr.Plot(label="Temperature History")
                
            with gr.Column(scale=2):
                gr.Markdown("###  Chat Interface")
                
                prompt_input = gr.Textbox(
                    lines=3,
                    placeholder="Enter ANY prompt here...\n• Ask factual questions: 'What is the capital of France?'\n• Request creative writing: 'Write a story about a robot falling in love'\n• Seek advice: 'How to improve my productivity?'",
                    label="Your Prompt"
                )
                
                with gr.Row():
                    analyze_btn = gr.Button(" Analyze Prompt", variant="primary")
                    generate_btn = gr.Button(" Generate Response", variant="primary")
                
                gr.Markdown("---")
                
                analysis_output = gr.Markdown(
                    label="Analysis Results",
                    value="** Analysis Results**\n\n*Your prompt analysis will appear here...*"
                )
                
                with gr.Row():
                    current_temp = gr.Number(
                        label="Current Temperature", 
                        value=0.7,
                        precision=3
                    )
                    similarity_chart = gr.Plot(label="Category Similarities")
                
                gr.Markdown("---")
                response_output = gr.Markdown(
                    label="LLM Response",
                    value="**💡 LLM Response**\n\n*Generated response will appear here...*"
                )
                
                gr.Markdown("---")
                history_output = gr.Markdown(
                    label="Conversation History",
                    value=get_conversation_history()
                )
        
        # Event handlers
        analyze_btn.click(
            analyze_prompt,
            [prompt_input, base_temp, scale_factor],
            [analysis_output, current_temp, similarity_chart, gr.State()]
        )
        
        generate_btn.click(
            generate_response,
            [prompt_input, base_temp, scale_factor, use_smart_temp],
            [response_output, current_temp, temp_chart]
        ).then(
            get_conversation_history,
            [],
            [history_output]
        )
        
        clear_btn.click(
            clear_history,
            [],
            [history_output, temp_chart]
        )
        
        # Examples
        gr.Markdown("### Example Prompts")
        examples = gr.Examples(
            examples=[
                "What is the capital of Brazil and its population?",
                "Write a short story about a robot learning to love",
                "How do I make chocolate chip cookies from scratch?",
                "Explain quantum computing in simple terms",
                "What are the ethical implications of artificial intelligence?",
                "Give me advice on improving my public speaking skills"
            ],
            inputs=prompt_input,
            label="Click any example to try it!"
        )
    
    return demo

if __name__ == "__main__":
    print("Starting Simple SmartTemp Engine...")
    print(" Dynamic Temperature Adjustment System")
    print("=" * 50)
    print("Loading interface...")
    
    try:
        demo = create_simple_interface()
        print("✅ Interface created successfully!")
        print("🌐 Starting server...")
        print("📍 Your app will be available at: http://localhost:7860")
        print(" Press Ctrl+C to stop the server")
        
        # Launch with better configuration
        demo.launch(
            server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
            server_port=7860,
            share=False,
            show_error=True,
            inbrowser=True,  # Try to open browser automatically
            quiet=False,     # Show more logs
            debug=False
        )
        
    except Exception as e:
        print(f" Error starting server: {e}")
        print(" Try these troubleshooting steps:")
        print("   1. Check if port 7860 is already in use")
        print("   2. Try running on a different port: demo.launch(server_port=7861)")
        print("   3. Make sure your firewall isn't blocking the connection")
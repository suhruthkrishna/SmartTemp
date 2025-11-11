import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import numpy as np
import os
import sys
import json

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smarttemp_engine import SmartTempEngine
from llm_integration import LLMIntegration

class SmartTempGradioApp:
    def __init__(self):
        self.engine = SmartTempEngine()
        self.llm = LLMIntegration()
        self.conversation_history = []
        self.temperature_history = []
        self.llm_available = self.llm.health_check()
        
        print(" SmartTemp Gradio App Initialized")
        print(f" LLM Available: {self.llm_available}")
        
    def analyze_prompt(self, prompt, base_temp, scale_factor):
        """Analyze prompt and return results"""
        if not prompt or not prompt.strip():
            return " Please enter a prompt", {}, 0.7, None, None
        
        # Update engine parameters
        self.engine.base_temp = base_temp
        self.engine.scale_factor = scale_factor
        
        try:
            # Analyze the prompt
            analysis = self.engine.analyze_prompt(prompt)
            
            # Store for history
            self.temperature_history.append({
                'timestamp': datetime.now(),
                'temperature': analysis['temperature'],
                'category': analysis['category'],
                'prompt': prompt[:50] + "..." if len(prompt) > 50 else prompt,
                'confidence': analysis['confidence']
            })
            
            # Create similarity chart
            similarities = analysis['all_similarities']
            df = pd.DataFrame(list(similarities.items()), columns=['Category', 'Similarity'])
            fig_similarity = px.bar(
                df, 
                x='Category', 
                y='Similarity', 
                color='Similarity',
                color_continuous_scale='viridis',
                title="🎯 Similarity to Each Category",
                labels={'Similarity': 'Confidence Score', 'Category': 'Prompt Category'}
            )
            fig_similarity.update_layout(
                yaxis_range=[0, 1],
                showlegend=False,
                height=400
            )
            
            # Create temperature history chart
            fig_temperature = self._create_temperature_chart()
            
            # Create analysis result text
            category_desc = self.engine.get_category_description(analysis['category'])
            temp_range = self.engine.get_optimal_temperature_range(analysis['category'])
            
            analysis_text = f"""
            ## Analysis Results

            ** Detected Category:** `{analysis['category'].title()}`
            ** Confidence Score:** `{analysis['confidence']:.3f}`
            ** Assigned Temperature:** `{analysis['temperature']:.3f}`

            ** Description:** {category_desc}
            ** Optimal Range:** `{temp_range[0]:.1f} - {temp_range[1]:.1f}`

            **Explanation:** The system analyzed your prompt and determined it's most similar to **{analysis['category']}** type with **{analysis['confidence']:.1%}** confidence. The temperature has been automatically set to **{analysis['temperature']:.3f}** for optimal response quality.
            """
            
            return analysis_text, analysis, analysis['temperature'], fig_similarity, fig_temperature
            
        except Exception as e:
            error_text = f" Error analyzing prompt: {str(e)}"
            return error_text, {}, base_temp, None, None
    
    def generate_response(self, prompt, base_temp, scale_factor, use_smart_temp):
        """Generate response with optional smart temperature"""
        if not prompt or not prompt.strip():
            return " Please enter a prompt", 0.7, None
        
        # Update engine parameters
        self.engine.base_temp = base_temp
        self.engine.scale_factor = scale_factor
        
        try:
            if use_smart_temp:
                # Use smart temperature - analyze prompt first
                analysis = self.engine.analyze_prompt(prompt)
                temperature = analysis['temperature']
                category_info = f"**Smart Analysis:** `{analysis['category'].title()}` | **📈 Confidence:** `{analysis['confidence']:.3f}` | **🌡️ Temperature:** `{temperature:.3f}`\n\n"
                category = analysis['category']
            else:
                # Use base temperature directly
                temperature = base_temp
                category_info = f"** Fixed Temperature:** `{temperature:.3f}`\n\n"
                category = 'fixed'
            
            # Generate response
            start_time = time.time()
            response = self.llm.generate_response(prompt, temperature)
            response_time = time.time() - start_time
            
            # Add generation info to response
            generation_info = f"\n\n---\n*⏱️ Generated in {response_time:.2f}s with temperature {temperature:.3f}*"
            full_response = response + generation_info
            
            # Store in history
            conversation_entry = {
                'timestamp': datetime.now(),
                'prompt': prompt,
                'response': full_response,
                'temperature': temperature,
                'category': category,
                'response_time': response_time
            }
            self.conversation_history.append(conversation_entry)
            
            # Update temperature history
            self.temperature_history.append({
                'timestamp': datetime.now(),
                'temperature': temperature,
                'category': category,
                'prompt': prompt[:50] + "..." if len(prompt) > 50 else prompt,
                'confidence': analysis['confidence'] if use_smart_temp else 1.0
            })
            
            # Create full response text with category info
            final_response = f"{category_info}{full_response}"
            
            # Update temperature chart
            fig_temperature = self._create_temperature_chart()
            
            return final_response, temperature, fig_temperature
            
        except Exception as e:
            error_msg = f" Error generating response: {str(e)}"
            return error_msg, base_temp, None
    
    def _create_temperature_chart(self):
        """Create temperature history chart"""
        if not self.temperature_history:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No temperature data yet.<br>Generate some responses to see the chart!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=" Temperature Adjustment History",
                xaxis_title="Time",
                yaxis_title="Temperature",
                yaxis_range=[0, 1],
                height=400
            )
            return fig
        
        df = pd.DataFrame(self.temperature_history)
        if len(df) > 0:
            df['time'] = df['timestamp'].dt.strftime('%H:%M:%S')
            
            fig = px.line(
                df, 
                x='time', 
                y='temperature', 
                color='category',
                title=' Temperature Adjustment History',
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Set2,
                hover_data=['prompt', 'confidence']
            )
            fig.update_layout(
                yaxis_range=[0, 1], 
                xaxis_title='Time',
                yaxis_title='Temperature',
                showlegend=True,
                hovermode='x unified',
                height=400
            )
            return fig
        return None
    
    def get_conversation_history(self):
        """Get formatted conversation history"""
        if not self.conversation_history:
            return "## Conversation History\n\nNo conversations yet. Start by generating some responses!"
        
        history_text = "## Conversation History\n\n"
        for i, entry in enumerate(reversed(self.conversation_history[-5:]), 1):
            idx = len(self.conversation_history) - i + 1
            history_text += f"### Conversation {idx}\n"
            history_text += f"**Time:** {entry['timestamp'].strftime('%H:%M:%S')}\n"
            history_text += f"**Category:** {entry['category'].title()}\n"
            history_text += f"**Temperature:** {entry['temperature']:.3f}\n"
            history_text += f"**Response Time:** {entry['response_time']:.2f}s\n"
            history_text += f"**Prompt:** {entry['prompt']}\n"
            history_text += f"**Response:** {entry['response'][:200]}...\n\n"
            history_text += "---\n\n"
        
        return history_text
    
    def get_analytics(self):
        """Get analytics data"""
        if not self.temperature_history:
            return "## Analytics\n\nNo data yet. Generate some responses to see analytics!"
        
        df = pd.DataFrame(self.temperature_history)
        
        analytics_text = "## Analytics\n\n"
        
        # Category distribution
        category_counts = df['category'].value_counts()
        analytics_text += "### Category Distribution\n"
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            analytics_text += f"- **{category.title()}:** {count} prompts ({percentage:.1f}%)\n"
        
        # Temperature stats
        analytics_text += f"\n### Temperature Statistics\n"
        analytics_text += f"- **Average Temperature:** `{df['temperature'].mean():.3f}`\n"
        analytics_text += f"- **Min Temperature:** `{df['temperature'].min():.3f}`\n"
        analytics_text += f"- **Max Temperature:** `{df['temperature'].max():.3f}`\n"
        analytics_text += f"- **Temperature Std Dev:** `{df['temperature'].std():.3f}`\n"
        
        # Confidence stats (if available)
        if 'confidence' in df.columns:
            conf_df = df[df['confidence'] < 1.0]  # Exclude fixed temperature entries
            if len(conf_df) > 0:
                analytics_text += f"\n### Confidence Statistics\n"
                analytics_text += f"- **Average Confidence:** `{conf_df['confidence'].mean():.3f}`\n"
                analytics_text += f"- **Min Confidence:** `{conf_df['confidence'].min():.3f}`\n"
                analytics_text += f"- **Max Confidence:** `{conf_df['confidence'].max():.3f}`\n"
        
        analytics_text += f"\n### Total Data Points\n"
        analytics_text += f"- **Total Prompts Analyzed:** {len(df)}\n"
        analytics_text += f"- **LLM Status:** {'Available' if self.llm_available else '❌ Using Mock'}\n"
        
        return analytics_text
    
    def clear_history(self):
        """Clear all history"""
        self.conversation_history.clear()
        self.temperature_history.clear()
        return "History cleared!", None, "## Conversation History\n\nHistory cleared! Start new conversations to see them here."

def create_gradio_interface():
    """Create the Gradio interface"""
    app = SmartTempGradioApp()
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple"
        ),
        title="SmartTemp LLM Engine",
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .example-prompt {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .example-prompt:hover {
            background-color: #f5f5f5;
            border-color: #2196F3;
        }
        """
    ) as demo:
        gr.Markdown("""
        # SmartTemp LLM Engine
        
        **Dynamically adjusts LLM temperature based on prompt type in real-time**
        
        *Enter any prompt and watch the system automatically adjust temperature for optimal responses!*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Configuration")
                
                with gr.Group():
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
                    value=True, 
                    label=" Use Smart Temperature",
                    info="Dynamically adjust temperature based on prompt analysis"
                )
                
                gr.Markdown("---")
                
                with gr.Group():
                    with gr.Row():
                        clear_btn = gr.Button("Clear History", variant="secondary", size="sm")
                        analytics_btn = gr.Button("Show Analytics", variant="secondary", size="sm")
                
                gr.Markdown("---")
                gr.Markdown("## Live Charts")
                
                temp_chart = gr.Plot(
                    label="Temperature History",
                    show_label=True
                )
                
            with gr.Column(scale=2):
                gr.Markdown("## Chat Interface")
                
                with gr.Group():
                    prompt_input = gr.Textbox(
                        lines=3,
                        placeholder="Enter ANY prompt here...\n• Ask factual questions: 'What is the capital of France?'\n• Request creative writing: 'Write a story about a robot falling in love'\n• Seek advice: 'How to improve my productivity?'\n• Analytical questions: 'Compare machine learning and deep learning'",
                        label="Your Prompt",
                        show_copy_button=True
                    )
                
                with gr.Row():
                    analyze_btn = gr.Button("Analyze Prompt", variant="primary", size="lg")
                    generate_btn = gr.Button("Generate Response", variant="primary", size="lg")
                
                gr.Markdown("---")
                
                with gr.Group():
                    analysis_output = gr.Markdown(
                        label="Analysis Results",
                        value="**Analysis Results**\n\n*Your prompt analysis will appear here...*"
                    )
                    
                    with gr.Row():
                        current_temp = gr.Number(
                            label="Current Temperature", 
                            value=0.7,
                            precision=3,
                            interactive=False
                        )
                        similarity_chart = gr.Plot(
                            label="Category Similarities",
                            show_label=True
                        )
                
                gr.Markdown("---")
                
                with gr.Group():
                    response_output = gr.Markdown(
                        label="LLM Response",
                        value="**LLM Response**\n\n*Generated response will appear here...*"
                    )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=2):
                history_output = gr.Markdown(
                    label="Conversation History",
                    value=app.get_conversation_history()
                )
            with gr.Column(scale=1):
                analytics_output = gr.Markdown(
                    label="Analytics",
                    value="## 📈 Analytics\n\n*Click 'Show Analytics' to view statistics*"
                )
        
        gr.Markdown("---")
        
        gr.Markdown("## Example Prompts")
        
        examples = gr.Examples(
            examples=[
                "What is the capital of Brazil and its population?",
                "Write a short story about a robot learning to love",
                "How do I make chocolate chip cookies from scratch?",
                "Explain quantum computing in simple terms",
                "What are the ethical implications of artificial intelligence?",
                "Give me advice on improving my public speaking skills",
                "Compare Python and JavaScript for web development",
                "What is the meaning of life according to different philosophies?",
                "Create a business plan for a new tech startup",
                "How does the human immune system work?"
            ],
            inputs=prompt_input,
            label="Click any example to try it!",
            examples_per_page=6
        )
        
        # Event handlers
        analyze_btn.click(
            fn=app.analyze_prompt,
            inputs=[prompt_input, base_temp, scale_factor],
            outputs=[analysis_output, gr.State(), current_temp, similarity_chart, temp_chart]
        )
        
        generate_btn.click(
            fn=app.generate_response,
            inputs=[prompt_input, base_temp, scale_factor, use_smart_temp],
            outputs=[response_output, current_temp, temp_chart]
        ).then(
            fn=app.get_conversation_history,
            inputs=[],
            outputs=[history_output]
        )
        
        clear_btn.click(
            fn=app.clear_history,
            inputs=[],
            outputs=[history_output, temp_chart, analytics_output]
        )
        
        analytics_btn.click(
            fn=app.get_analytics,
            inputs=[],
            outputs=[analytics_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True
    )
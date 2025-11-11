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

# Try different import approaches
try:
    # First try: import from current package
    from .smarttemp_engine import SmartTempEngine
    from .llm_integration import LLMIntegration
    print("✅ Imported using relative imports")
except ImportError:
    try:
        # Second try: import directly from src
        from smarttemp_engine import SmartTempEngine
        from llm_integration import LLMIntegration
        print("✅ Imported using direct imports")
    except ImportError:
        try:
            # Third try: add parent directory to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from src.smarttemp_engine import SmartTempEngine
            from src.llm_integration import LLMIntegration
            print("✅ Imported using path manipulation")
        except ImportError as e:
            print(f"❌ All import attempts failed: {e}")
            # Fallback: define minimal versions
            print("🔄 Using fallback implementations...")
            
            # Minimal fallback implementation
            class FallbackSmartTempEngine:
                def __init__(self, base_temp=0.7, scale_factor=0.3):
                    self.base_temp = base_temp
                    self.scale_factor = scale_factor
                
                def analyze_prompt(self, prompt):
                    # Simple fallback analysis
                    prompt_lower = prompt.lower()
                    if any(word in prompt_lower for word in ['what', 'when', 'where', 'who', 'capital', 'population']):
                        category = 'factual'
                        confidence = 0.9
                        temperature = 0.1
                    elif any(word in prompt_lower for word in ['write', 'story', 'poem', 'creative', 'imagine']):
                        category = 'creative'
                        confidence = 0.9
                        temperature = 0.9
                    elif any(word in prompt_lower for word in ['how', 'make', 'cook', 'step', 'instructions']):
                        category = 'instructional'
                        confidence = 0.8
                        temperature = 0.3
                    elif any(word in prompt_lower for word in ['compare', 'analyze', 'explain', 'difference']):
                        category = 'analytical'
                        confidence = 0.7
                        temperature = 0.5
                    elif any(word in prompt_lower for word in ['advice', 'should i', 'help me', 'improve']):
                        category = 'personal'
                        confidence = 0.6
                        temperature = 0.6
                    else:
                        category = 'analytical'
                        confidence = 0.5
                        temperature = 0.7
                    
                    return {
                        'category': category,
                        'confidence': confidence,
                        'temperature': temperature,
                        'all_similarities': {category: confidence},
                        'prompt': prompt
                    }
            
            class FallbackLLMIntegration:
                def generate_response(self, prompt, temperature, max_tokens=500):
                    responses = {
                        0.1: f"**Factual Response** (Temp: {temperature})\n\nBased on your query: '{prompt}'\n\nKey facts: [Factual information would go here]\n\nThis low-temperature response focuses on accuracy and precision.",
                        0.3: f"**Instructional Response** (Temp: {temperature})\n\nRegarding '{prompt}':\n\nStep 1: First step\nStep 2: Second step\nStep 3: Final step\n\nThis medium-low temperature provides clear instructions.",
                        0.5: f"**Analytical Response** (Temp: {temperature})\n\nAnalysis of '{prompt}':\n\nMultiple perspectives exist. On one hand... On the other hand... The evidence suggests a balanced approach.\n\nMedium temperature enables comprehensive analysis.",
                        0.7: f"**Exploratory Response** (Temp: {temperature})\n\nExploring '{prompt}' reveals interesting dimensions. There are several ways to approach this topic, each with unique considerations and potential outcomes worth contemplating.\n\nHigher temperature allows more exploratory thinking.",
                        0.9: f"**Creative Response** (Temp: {temperature})\n\nAh, '{prompt}' - what an imaginative concept! Let's venture beyond conventional boundaries and explore the realms of possibility where new ideas blossom like spring flowers.\n\nHigh temperature enables maximum creativity!"
                    }
                    
                    # Find closest temperature
                    closest_temp = min(responses.keys(), key=lambda x: abs(x - temperature))
                    return responses[closest_temp]
            
            SmartTempEngine = FallbackSmartTempEngine
            LLMIntegration = FallbackLLMIntegration
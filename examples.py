#!/usr/bin/env python3
"""
Example usage and testing for SmartTemp LLM Engine
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from smarttemp_engine import SmartTempEngine
from llm_integration import LLMIntegration

def run_examples():
    """Run comprehensive examples of the SmartTemp system"""
    
    print(" SmartTemp LLM Engine - Comprehensive Examples")
    print("=" * 60)
    
    # Initialize components
    engine = SmartTempEngine()
    llm = LLMIntegration()
    
    # Test prompts covering all categories
    test_cases = [
        {
            "prompt": "What is the population of Tokyo, Japan as of 2024?",
            "expected_category": "factual",
            "description": "Factual query seeking specific data"
        },
        {
            "prompt": "Write a creative short story about a robot who becomes an artist",
            "expected_category": "creative", 
            "description": "Creative writing request"
        },
        {
            "prompt": "How do I change a flat tire on a car step by step?",
            "expected_category": "instructional",
            "description": "Step-by-step instructions"
        },
        {
            "prompt": "Compare and contrast machine learning with traditional programming approaches",
            "expected_category": "analytical",
            "description": "Analytical comparison"
        },
        {
            "prompt": "What is the meaning of happiness according to different philosophical traditions?",
            "expected_category": "philosophical", 
            "description": "Philosophical exploration"
        },
        {
            "prompt": "How can I improve my time management skills for better productivity?",
            "expected_category": "personal",
            "description": "Personal advice request"
        }
    ]
    
    print("\n Testing Prompt Analysis:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        prompt = test_case["prompt"]
        expected = test_case["expected_category"]
        description = test_case["description"]
        
        print(f"\n{i}. {description}")
        print(f"   Prompt: '{prompt}'")
        
        # Analyze prompt
        analysis = engine.analyze_prompt(prompt)
        
        print(f"   Detected: {analysis['category']} (expected: {expected})")
        print(f"    Confidence: {analysis['confidence']:.3f}")
        print(f"    Temperature: {analysis['temperature']:.3f}")
        
        # Show top 2 similarities
        similarities = analysis['all_similarities']
        top_categories = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:2]
        print(f"    Top matches: {', '.join([f'{cat}({sim:.3f})' for cat, sim in top_categories])}")
    
    print("\n" + "=" * 60)
    print(" Testing LLM Integration with Different Temperatures:")
    print("-" * 50)
    
    # Test LLM with different temperatures
    test_prompt = "Explain the concept of artificial intelligence"
    
    for temp in [0.1, 0.5, 0.9]:
        print(f"\n Temperature {temp:.1f}:")
        response = llm.generate_response(test_prompt, temp, max_tokens=150)
        if response:
            preview = response[:100] + "..." if len(response) > 100 else response
            print(f"   Response: {preview}")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("\n Try running the web interface for interactive usage:")
    print("   python run_gradio.py    # For Gradio interface")
    print("   python run_streamlit.py # For Streamlit interface")

if __name__ == "__main__":
    run_examples()
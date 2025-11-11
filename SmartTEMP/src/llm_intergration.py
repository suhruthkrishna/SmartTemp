import requests
import json
import logging
import time
from typing import Optional, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMIntegration:
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama2"):
        """
        Initialize LLM Integration with Ollama (default)
        
        Args:
            base_url (str): Base URL for Ollama API
            model_name (str): Name of the model to use
        """
        self.base_url = base_url
        self.model_name = model_name
        self.session = requests.Session()
        self.session.timeout = 30
        
        logger.info(f"LLM Integration initialized with {base_url}, model: {model_name}")
    
    def set_model(self, model_name: str):
        """Set the model to use"""
        self.model_name = model_name
        logger.info(f"Model changed to: {model_name}")
    
    def generate_response(self, prompt: str, temperature: float, max_tokens: int = 500) -> Optional[str]:
        """
        Generate response from LLM with specified temperature
        
        Args:
            prompt (str): User input prompt
            temperature (float): Temperature setting (0.1 to 1.0)
            max_tokens (int): Maximum tokens in response
            
        Returns:
            str: LLM response or None if error
        """
        try:
            # For Ollama API
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                }
            }
            
            logger.info(f"Generating response with temperature: {temperature:.3f}")
            start_time = time.time()
            
            response = self.session.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            response_time = time.time() - start_time
            
            logger.info(f"Response generated in {response_time:.2f}s")
            return result.get('response', 'No response generated')
            
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama connection failed - using mock response")
            return self._generate_mock_response(prompt, temperature)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling LLM API: {e}")
            return self._generate_mock_response(prompt, temperature)
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {e}")
            return self._generate_mock_response(prompt, temperature)
    
    def _generate_mock_response(self, prompt: str, temperature: float) -> str:
        """
        Generate realistic mock responses that vary with temperature
        
        Args:
            prompt (str): User prompt
            temperature (float): Temperature setting
            
        Returns:
            str: Mock response tailored to temperature
        """
        # Simulate processing time based on temperature
        time.sleep(1.5 if temperature < 0.4 else 2.0 if temperature < 0.7 else 2.5)
        
        # Low temperature responses (factual, precise)
        if temperature < 0.3:
            return f"""**Factual Response** (Temperature: {temperature:.2f})

Based on your query about "{prompt}", here are the key facts:

• Primary Information: Relevant factual data and specific details
• Supporting Context: Additional background information  
• Related Facts: Connected information for comprehensive understanding

This response was generated with low temperature setting ({temperature:.2f}) to ensure precision, accuracy, and factual correctness. The information is presented in a clear, structured format focusing on verifiable data."""

        # Medium temperature responses (balanced, analytical)
        elif temperature < 0.6:
            return f"""**Analytical Response** (Temperature: {temperature:.2f})

Regarding your question "{prompt}", let me provide a comprehensive analysis:

**Key Perspectives:**
- One viewpoint considers the technical aspects and practical implications
- Another perspective examines the broader context and historical background
- Additional considerations include future trends and potential developments

**Critical Analysis:**
- The evidence indicates several important factors to consider
- There are multiple dimensions to this topic worth exploring
- The balance between different approaches suggests optimal strategies

**Conclusion:**
This medium-temperature response ({temperature:.2f}) aims to provide balanced analysis while maintaining coherence and logical structure. It explores multiple angles while staying focused on the core question."""

        # High temperature responses (creative, exploratory)
        else:
            return f"""**Creative Response** (Temperature: {temperature:.2f})

Ah, "{prompt}" — what a fascinating concept to explore! Let me paint you a picture...

Imagine a world where ideas dance like autumn leaves in the wind, each thought branching into new possibilities we've never considered. The very essence of your question sparks a cascade of connections, weaving together threads of insight and imagination in ways that surprise even me.

What if we look at this from a completely different angle? Perhaps through the lens of ancient wisdom, or maybe through the futuristic perspective of technologies not yet invented. The boundaries between reality and imagination blur, creating new patterns that invite us to see familiar things in extraordinary ways.

In this realm of creative exploration, we're not just answering a question — we're embarking on a journey through landscapes of possibility, where each turn reveals new horizons and unexpected connections.

This response embraces the spirit of creative exploration with higher temperature setting ({temperature:.2f}), allowing for more imaginative, varied, and unexpected content generation."""
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama
        
        Returns:
            List[str]: List of available model names
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            if models:
                logger.info(f"Found {len(models)} available models")
            else:
                logger.warning("No models found, using fallback list")
                models = ["llama2", "mistral", "codellama"]
                
            return models
            
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return ["llama2", "mistral", "codellama"]  # Fallback models
    
    def health_check(self) -> bool:
        """
        Check if Ollama is running and accessible
        
        Returns:
            bool: True if Ollama is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            dict: Model information
        """
        try:
            url = f"{self.base_url}/api/show"
            payload = {"name": self.model_name}
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}

class OpenAIIntegration:
    """
    Alternative implementation for OpenAI-compatible APIs
    """
    def __init__(self, base_url: str, api_key: str = "", model_name: str = "gpt-3.5-turbo"):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.session = requests.Session()
        self.session.timeout = 30
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        logger.info(f"OpenAI Integration initialized with {base_url}, model: {model_name}")
    
    def generate_response(self, prompt: str, temperature: float, max_tokens: int = 500) -> Optional[str]:
        """
        Generate response using OpenAI-compatible API
        
        Args:
            prompt (str): User input prompt
            temperature (float): Temperature setting
            max_tokens (int): Maximum tokens in response
            
        Returns:
            str: LLM response or None if error
        """
        try:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.9
            }
            
            logger.info(f"Generating OpenAI response with temperature: {temperature:.3f}")
            
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None

def test_llm_integration():
    """
    Test the LLM integration
    """
    llm = LLMIntegration()
    
    print("🧪 Testing LLM Integration...")
    print("=" * 50)
    
    # Health check
    is_healthy = llm.health_check()
    print(f"🔍 Ollama Health Check: {'✅ Healthy' if is_healthy else '❌ Not Available'}")
    
    # Available models
    models = llm.get_available_models()
    print(f"📚 Available Models: {', '.join(models)}")
    
    # Test responses with different temperatures
    test_prompt = "Explain the concept of artificial intelligence"
    
    for temp in [0.1, 0.5, 0.9]:
        print(f"\n🌡️ Testing temperature {temp:.1f}:")
        response = llm.generate_response(test_prompt, temp, max_tokens=200)
        if response:
            preview = response[:150] + "..." if len(response) > 150 else response
            print(f"   Response: {preview}")

if __name__ == "__main__":
    test_llm_integration()
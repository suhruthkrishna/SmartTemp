import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartTempEngine:
    def __init__(self, base_temp=0.7, scale_factor=0.3):
        """
        Initialize the SmartTemp Engine for real-time prompt analysis
        """
        self.base_temp = base_temp
        self.scale_factor = scale_factor
        
        # Initialize the embedding model
        logger.info("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Define category descriptions for real-time analysis
        self.category_descriptions = {
            'factual': "Questions seeking specific facts, data, definitions, or concrete information. Examples: what, when, where, who questions about verifiable information.",
            'analytical': "Requests for comparison, analysis, explanation of processes, or breaking down complex topics. Examples: compare, analyze, explain how, pros and cons.",
            'creative': "Prompts requesting original content, stories, poems, ideas, or imaginative scenarios. Examples: write a story, create, imagine, generate ideas.",
            'philosophical': "Questions about meaning, ethics, consciousness, abstract concepts, or deep reasoning. Examples: why, meaning of, ethical implications, philosophical questions.",
            'personal': "Requests for advice, personal development, opinions, or subjective guidance. Examples: how to improve, advice for, what should I do, personal growth.",
            'instructional': "Step-by-step guides, tutorials, recipes, or procedural information. Examples: how to make, steps to, tutorial, guide to."
        }
        
        # Pre-compute embeddings for category descriptions
        self.category_embeddings = {}
        for category, description in self.category_descriptions.items():
            embedding = self.embedding_model.encode([description])[0]
            self.category_embeddings[category] = embedding
        
        logger.info("SmartTemp Engine initialized successfully")

    def analyze_prompt(self, prompt):
        """
        Analyze ANY prompt in real-time and determine its category
        
        Args:
            prompt (str): User input prompt
            
        Returns:
            dict: Analysis results including category, confidence, temperature, and similarities
        """
        if not prompt or not prompt.strip():
            return {
                'category': 'analytical',
                'confidence': 0.5,
                'temperature': self.base_temp,
                'all_similarities': {},
                'prompt': prompt
            }
        
        try:
            # Encode the input prompt
            prompt_embedding = self.embedding_model.encode([prompt])[0]
            
            # Calculate similarity with each category description
            similarities = {}
            for category, category_embedding in self.category_embeddings.items():
                similarity = cosine_similarity(
                    [prompt_embedding], 
                    [category_embedding]
                )[0][0]
                similarities[category] = similarity
            
            # Find the best matching category
            best_category = max(similarities.items(), key=lambda x: x[1])
            category, confidence = best_category
            
            # Calculate dynamic temperature
            temperature = self.calculate_temperature(confidence, category)
            
            return {
                'category': category,
                'confidence': float(confidence),
                'temperature': float(temperature),
                'all_similarities': similarities,
                'prompt': prompt
            }
            
        except Exception as e:
            logger.error(f"Error analyzing prompt: {e}")
            return {
                'category': 'analytical',
                'confidence': 0.5,
                'temperature': self.base_temp,
                'all_similarities': {},
                'prompt': prompt,
                'error': str(e)
            }
    
    def calculate_temperature(self, confidence, category):
        """
        Calculate dynamic temperature based on confidence and category
        
        Args:
            confidence (float): Confidence score from 0 to 1
            category (str): Detected prompt category
            
        Returns:
            float: Calculated temperature value between 0.1 and 1.0
        """
        # Define optimal base temperatures for each category
        category_base_temps = {
            'factual': 0.1,      # Low temp for accuracy and precision
            'instructional': 0.3, # Medium-low for clear, step-by-step instructions
            'analytical': 0.5,    # Balanced for reasoned analysis
            'personal': 0.6,      # Some creativity in advice and opinions
            'philosophical': 0.7, # Exploratory for abstract concepts
            'creative': 0.9       # High creativity for imaginative content
        }
        
        # Get base temperature for the detected category
        category_base_temp = category_base_temps.get(category, self.base_temp)
        
        # Adjust based on confidence using the original formula with enhancements
        if confidence > 0.7:  # High confidence - use category-optimal temperature
            temperature = category_base_temp
        elif confidence > 0.4:  # Medium confidence - blend between category temp and exploratory temp
            # Original formula: temperature = base_temp + (1 - confidence) * scale_factor
            # Enhanced: Blend between category optimal and higher exploratory temperature
            blend_factor = (0.7 - confidence) / 0.3  # 0 to 1 scale
            temperature = category_base_temp + (0.8 - category_base_temp) * blend_factor
        else:  # Low confidence - use high temperature for exploration
            temperature = 0.8
        
        # Clamp between 0.1 and 1.0 as specified in the original requirement
        return max(0.1, min(1.0, temperature))
    
    def get_category_description(self, category):
        """
        Get human-readable description for each category
        
        Args:
            category (str): Category name
            
        Returns:
            str: Description of the category
        """
        descriptions = {
            'factual': 'Precise, factual information with low temperature for accuracy',
            'analytical': 'Balanced analysis with moderate temperature',
            'creative': 'Imaginative and exploratory with high temperature',
            'philosophical': 'Thoughtful exploration with medium-high temperature',
            'personal': 'Empathic and contextual with medium temperature',
            'instructional': 'Clear, step-by-step with medium-low temperature'
        }
        return descriptions.get(category, 'General purpose response')
    
    def get_optimal_temperature_range(self, category):
        """
        Get the optimal temperature range for each category
        
        Args:
            category (str): Category name
            
        Returns:
            tuple: (min_temp, max_temp) range
        """
        ranges = {
            'factual': (0.1, 0.3),
            'instructional': (0.2, 0.4),
            'analytical': (0.4, 0.6),
            'personal': (0.5, 0.7),
            'philosophical': (0.6, 0.8),
            'creative': (0.7, 1.0)
        }
        return ranges.get(category, (0.3, 0.7))

def test_engine():
    """
    Test the SmartTemp engine with various example prompts
    """
    engine = SmartTempEngine()
    
    test_prompts = [
        "What is the capital of Japan and its population?",
        "Write a poem about artificial intelligence and human emotions",
        "How do I learn programming effectively as a beginner?",
        "What are the ethical implications of advanced AI systems?",
        "Explain quantum computing to a 10-year-old child",
        "How to make perfect chocolate chip cookies from scratch?",
        "Compare and contrast machine learning with traditional programming",
        "What is the meaning of life according to different philosophical traditions?",
        "Give me advice on improving my public speaking skills",
        "Create a short story about a time-traveling historian"
    ]
    
    print("🧪 Testing SmartTemp Engine with various prompts...")
    print("=" * 80)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. 📝 Prompt: {prompt}")
        result = engine.analyze_prompt(prompt)
        print(f"    Category: {result['category'].title()}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"    Temperature: {result['temperature']:.3f}")
        
        # Show top 3 similarities
        similarities = result['all_similarities']
        top_categories = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"    Top similarities: {', '.join([f'{cat}({sim:.3f})' for cat, sim in top_categories])}")
        
        print("   " + "-" * 50)

if __name__ == "__main__":
    test_engine()
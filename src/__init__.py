"""
SmartTemp LLM Engine - Dynamic temperature adjustment for LLMs based on prompt type
"""

__version__ = "1.0.0"
__author__ = "SmartTemp Team"
__description__ = "Dynamically adjusts LLM temperature based on prompt analysis"

from .smarttemp_engine import SmartTempEngine
from .llm_integration import LLMIntegration, OpenAIIntegration

__all__ = [
    "SmartTempEngine",
    "LLMIntegration", 
    "OpenAIIntegration"
]
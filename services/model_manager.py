"""
Model Manager - Handles fallback models when API quotas are exceeded
"""

import google.generativeai as genai
from typing import Optional, List, Any
import time


class ModelManager:
    """
    Manages model fallback logic for Gemini API.
    Automatically tries alternative models when quota is exceeded.
    """

    def __init__(self, api_key: str, model_list: List[str], retry_delay: float = 1.0):
        """
        Initialize model manager.

        Args:
            api_key: Google Gemini API key
            model_list: List of model names to try in order
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.model_list = model_list
        self.retry_delay = retry_delay
        self.current_model_index = 0
        self.model_cache = {}  # Cache model instances

        genai.configure(api_key=api_key)

    def get_model(self, model_name: Optional[str] = None):
        """
        Get a GenerativeModel instance.

        Args:
            model_name: Specific model name, or None to use current fallback model

        Returns:
            GenerativeModel instance
        """
        if model_name is None:
            model_name = self.model_list[self.current_model_index]

        # Return cached model if exists
        if model_name in self.model_cache:
            return self.model_cache[model_name]

        # Create new model instance
        model = genai.GenerativeModel(model_name)
        self.model_cache[model_name] = model
        return model

    def generate_with_fallback(self, prompt: str, **kwargs) -> str:
        """
        Generate content with automatic fallback to alternative models on quota errors.

        Args:
            prompt: The prompt to generate content from
            **kwargs: Additional arguments for generate_content()

        Returns:
            Generated text response

        Raises:
            Exception: If all fallback models fail
        """
        last_error = None

        for attempt, model_name in enumerate(self.model_list):
            try:
                print(f"[ModelManager] Attempting with model: {model_name}")
                model = self.get_model(model_name)
                response = model.generate_content(prompt, **kwargs)

                # Success - update current model index and return
                self.current_model_index = attempt
                print(f"[ModelManager] Success with model: {model_name}")
                return response.text

            except Exception as e:
                error_str = str(e)
                print(f"[ModelManager] Error with {model_name}: {error_str[:200]}")

                # Check if it's a quota error
                if "429" in error_str or "quota" in error_str.lower() or "resource_exhausted" in error_str.lower():
                    print(f"[ModelManager] Quota exceeded for {model_name}, trying next model...")
                    last_error = e

                    # Sleep before trying next model
                    if attempt < len(self.model_list) - 1:
                        time.sleep(self.retry_delay)
                    continue
                else:
                    # Non-quota error - raise immediately
                    raise e

        # All models failed with quota errors
        error_msg = f"All fallback models exhausted. Tried: {', '.join(self.model_list)}. Last error: {last_error}"
        print(f"[ModelManager] {error_msg}")
        raise Exception(error_msg)

    def reset_to_primary(self):
        """Reset to primary (first) model"""
        self.current_model_index = 0

    def get_current_model_name(self) -> str:
        """Get the name of the currently active model"""
        return self.model_list[self.current_model_index]

    def get_available_models(self) -> List[str]:
        """Get list of all configured models"""
        return self.model_list.copy()


class ChatModelManager:
    """
    Model manager for LangChain ChatGoogleGenerativeAI models.
    Provides fallback logic for chat-based models.
    """

    def __init__(self, api_key: str, model_list: List[str]):
        self.api_key = api_key
        self.model_list = model_list
        self.current_model_index = 0

    def get_chat_model(self, temperature: float = 0.3):
        """
        Get ChatGoogleGenerativeAI instance with current fallback model.

        Args:
            temperature: Model temperature setting

        Returns:
            ChatGoogleGenerativeAI instance
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("LangChain Google GenAI not installed. Run: pip install langchain-google-genai")

        model_name = self.model_list[self.current_model_index]
        print(f"[ChatModelManager] Using model: {model_name}")

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=temperature
        )

    def try_next_model(self) -> bool:
        """
        Switch to next fallback model.

        Returns:
            True if there's a next model, False if all models exhausted
        """
        if self.current_model_index < len(self.model_list) - 1:
            self.current_model_index += 1
            print(f"[ChatModelManager] Switching to fallback model: {self.model_list[self.current_model_index]}")
            return True
        return False

    def reset_to_primary(self):
        """Reset to primary (first) model"""
        self.current_model_index = 0

    def get_current_model_name(self) -> str:
        """Get the name of the currently active model"""
        return self.model_list[self.current_model_index]

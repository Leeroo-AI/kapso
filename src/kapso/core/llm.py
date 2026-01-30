# LLM Backend Module
#
# Unified LLM interface with support for completions, web search, and cost tracking.

import asyncio
import logging
import os
import time
from litellm import completion, acompletion
from typing import Optional, Dict, List

# Suppress verbose LiteLLM logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


class LLMBackend:
    """
    LLM backend with retry logic and cost tracking.
    
    Example:
        llm = LLMBackend()
        response = llm.llm_completion(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(llm.get_cumulative_cost())
    """
    
    def __init__(self):
        self._cumulative_cost = 0.0

    def get_cumulative_cost(self) -> float:
        """Returns cumulative cost of all LLM calls made by this instance."""
        return self._cumulative_cost
    
    def llm_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 1,
        reasoning_effort: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Call LLM with messages. Retries once after 2 min on failure.
        
        Args:
            model: Model name (e.g., "gpt-4.1-mini", "claude-sonnet-4-20250514")
            messages: List of message dicts with role and content
            temperature: Sampling temperature (default 1)
            reasoning_effort: Optional reasoning effort level
            
        Returns:
            Model response text
            
        Example:
            llm = LLMBackend()
            response = llm.llm_completion(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "What is 2+2?"}],
                temperature=0
            )
        """
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                drop_params=True,
                **kwargs
            )
            if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                self._cumulative_cost += response._hidden_params['response_cost']
            return response.choices[0].message.content
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("An error occured in calling LLMs, retrying in 2 minutes...")
            time.sleep(120)
            try:
                response = completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    drop_params=True,
                    **kwargs
                )
                if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                    self._cumulative_cost += response._hidden_params['response_cost']
                return response.choices[0].message.content
            except Exception as retry_e:
                raise Exception(f"Error calling model {model}: {str(retry_e)}")

    def llm_completion_with_system_prompt(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 1,
        reasoning_effort: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Call LLM with a system prompt and user message.
        
        Args:
            model: Model name
            system_prompt: System prompt content
            user_message: User message content
            temperature: Sampling temperature (default 1)
            reasoning_effort: Optional reasoning effort level
            
        Returns:
            Model response text
            
        Example:
            llm = LLMBackend()
            response = llm.llm_completion_with_system_prompt(
                model="gpt-4.1-mini",
                system_prompt="You are a helpful assistant.",
                user_message="Explain Python in one sentence."
            )
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self.llm_completion(model, messages, temperature, reasoning_effort=reasoning_effort, **kwargs)

    def llm_multiple_completions(
        self, models: List[str], messages: List[Dict[str, str]], temperature: float = 1, reasoning_effort: Optional[str] = None, **kwargs
    ) -> List[str]:
        """
        Call multiple models in parallel with the same messages.
        
        Args:
            models: List of model names to call
            messages: List of message dicts (same for all models)
            temperature: Sampling temperature (default 1)
            reasoning_effort: Optional reasoning effort level
            
        Returns:
            List of model response texts (same order as models)
            
        Example:
            llm = LLMBackend()
            responses = llm.llm_multiple_completions(
                models=["gpt-4.1-mini", "claude-sonnet-4-20250514"],
                messages=[{"role": "user", "content": "Say hello"}]
            )
        """
        async def _run():
            try:
                tasks = [
                    acompletion(
                        model=m, messages=messages, temperature=temperature, reasoning_effort=reasoning_effort, drop_params=True, **kwargs
                    ) for m in models
                ]
                results = await asyncio.gather(*tasks)
                for result in results:
                    if hasattr(result, '_hidden_params') and 'response_cost' in result._hidden_params:
                        self._cumulative_cost += result._hidden_params['response_cost']
                return [r.choices[0].message.content for r in results]
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print("An error occured in calling LLMs, retrying in 2 minutes...")
                await asyncio.sleep(120)
                try:
                    tasks = [
                        acompletion(
                            model=m, messages=messages, temperature=temperature, reasoning_effort=reasoning_effort, drop_params=True, **kwargs
                        ) for m in models
                    ]
                    results = await asyncio.gather(*tasks)
                    for result in results:
                        if hasattr(result, '_hidden_params') and 'response_cost' in result._hidden_params:
                            self._cumulative_cost += result._hidden_params['response_cost']
                    return [r.choices[0].message.content for r in results]
                except Exception as retry_e:
                    raise Exception(f"Error calling models {models}: {str(retry_e)}")
            
        return asyncio.run(_run())

    def llm_completion_with_web_search(
        self,
        model: str,
        messages: List[Dict[str, str]],
        search_context_size: str = "medium",
        reasoning_effort: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        LLM completion with web search enabled via LiteLLM.
        
        Uses LiteLLM's web_search_options for unified provider support.
        Supports: OpenAI, Gemini, Claude, Perplexity, xAI
        
        Args:
            model: Model name (will be mapped to search-enabled variant if needed)
            messages: List of message dicts with role and content
            search_context_size: "low", "medium", or "high"
            reasoning_effort: Optional reasoning effort level
            
        Returns:
            Model response text
            
        Note:
            Temperature is not passed to search-preview models as they don't support it.
            
        Example:
            llm = LLMBackend()
            response = llm.llm_completion_with_web_search(
                model="gpt-4o-search-preview",
                messages=[{"role": "user", "content": "What is today's date?"}],
                search_context_size="low"
            )
        """
        # Map models to their search-enabled variants
        search_model_map = {
            "gpt-5": "openai/gpt-4o-search-preview",
            "gpt-5.1": "openai/gpt-4o-search-preview",
            "gpt-5-mini": "openai/gpt-4o-search-preview",
            "gpt-4.1": "openai/gpt-4o-search-preview",
            "gpt-4.1-mini": "openai/gpt-4o-search-preview",
        }
        search_model = search_model_map.get(model, model)
        
        # Remove temperature from kwargs if present (search models don't support it)
        kwargs.pop('temperature', None)
        
        try:
            response = completion(
                model=search_model,
                messages=messages,
                reasoning_effort=reasoning_effort,
                web_search_options={"search_context_size": search_context_size},
                drop_params=True,
                **kwargs
            )
            if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                self._cumulative_cost += response._hidden_params['response_cost']
            return response.choices[0].message.content
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Web search completion error: {e}, retrying in 2 minutes...")
            time.sleep(120)
            try:
                response = completion(
                    model=search_model,
                    messages=messages,
                    reasoning_effort=reasoning_effort,
                    web_search_options={"search_context_size": search_context_size},
                    drop_params=True,
                    **kwargs
                )
                if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                    self._cumulative_cost += response._hidden_params['response_cost']
                return response.choices[0].message.content
            except Exception as retry_e:
                raise Exception(f"Error calling model {model} with web search: {str(retry_e)}")

    def llm_multiple_completions_with_web_search(
        self,
        models: List[str],
        messages: List[Dict[str, str]],
        search_context_size: str = "medium",
        reasoning_efforts: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Parallel LLM completions with web search enabled.
        
        Args:
            models: List of model names
            messages: List of message dicts
            search_context_size: "low", "medium", or "high"
            reasoning_efforts: Optional list of reasoning efforts per model
            
        Returns:
            List of model response texts
            
        Note:
            Temperature is not passed to search-preview models as they don't support it.
            
        Example:
            llm = LLMBackend()
            responses = llm.llm_multiple_completions_with_web_search(
                models=["gpt-4o-search-preview", "gpt-4o-search-preview"],
                messages=[{"role": "user", "content": "What is the capital of France?"}],
                search_context_size="low"
            )
        """
        # Map models to search-enabled variants
        search_model_map = {
            "gpt-5": "openai/gpt-4o-search-preview",
            "gpt-5.1": "openai/gpt-4o-search-preview",
            "gpt-5-mini": "openai/gpt-4o-search-preview",
            "gpt-4.1": "openai/gpt-4o-search-preview",
            "gpt-4.1-mini": "openai/gpt-4o-search-preview",
        }
        search_models = [search_model_map.get(m, m) for m in models]
        
        # Remove temperature from kwargs if present (search models don't support it)
        kwargs.pop('temperature', None)
        
        async def _run():
            try:
                tasks = []
                for i, m in enumerate(search_models):
                    effort = reasoning_efforts[i] if reasoning_efforts and i < len(reasoning_efforts) else None
                    tasks.append(
                        acompletion(
                            model=m,
                            messages=messages,
                            reasoning_effort=effort,
                            web_search_options={"search_context_size": search_context_size},
                            drop_params=True,
                            **kwargs
                        )
                    )
                results = await asyncio.gather(*tasks)
                for result in results:
                    if hasattr(result, '_hidden_params') and 'response_cost' in result._hidden_params:
                        self._cumulative_cost += result._hidden_params['response_cost']
                return [r.choices[0].message.content for r in results]
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Web search multiple completions error: {e}, retrying in 2 minutes...")
                await asyncio.sleep(120)
                try:
                    tasks = []
                    for i, m in enumerate(search_models):
                        effort = reasoning_efforts[i] if reasoning_efforts and i < len(reasoning_efforts) else None
                        tasks.append(
                            acompletion(
                                model=m,
                                messages=messages,
                                reasoning_effort=effort,
                                web_search_options={"search_context_size": search_context_size},
                                drop_params=True,
                                **kwargs
                            )
                        )
                    results = await asyncio.gather(*tasks)
                    for result in results:
                        if hasattr(result, '_hidden_params') and 'response_cost' in result._hidden_params:
                            self._cumulative_cost += result._hidden_params['response_cost']
                    return [r.choices[0].message.content for r in results]
                except Exception as retry_e:
                    raise Exception(f"Error calling models {models} with web search: {str(retry_e)}")
        
        return asyncio.run(_run())

    def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-large",
        max_chars: Optional[int] = None,
    ) -> List[float]:
        """
        Create embedding for text using OpenAI embeddings API.
        
        Args:
            text: Text to embed
            model: Embedding model name (default: "text-embedding-3-large")
            max_chars: Optional truncation limit.
                IMPORTANT: We do not hardcode truncation limits in code.
                If you need a limit due to upstream API constraints, pass it in
                explicitly or set `KAPSO_EMBEDDING_MAX_CHARS` in your `.env`.
            
        Returns:
            List of embedding floats, or empty list on error
        """
        try:
            import openai
            
            # Respect optional truncation limit from caller or environment.
            # Default is NO truncation (safer for correctness; callers can tune).
            if max_chars is None:
                env_val = os.getenv("KAPSO_EMBEDDING_MAX_CHARS")
                if env_val:
                    try:
                        parsed = int(env_val)
                        max_chars = parsed if parsed > 0 else None
                    except Exception:
                        max_chars = None
            
            input_text = text if (max_chars is None) else text[:max_chars]
            response = openai.embeddings.create(
                model=model,
                input=input_text,
            )
            return response.data[0].embedding
        except Exception:
            return []


def main():
    llm = LLMBackend()
    try:
        response = llm.llm_completion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
        )
        if response is None or response == "":
            print("Error: Received empty or None response")
        else:
            print(response)
        print(f"Cost: ${llm.get_cumulative_cost():.6f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

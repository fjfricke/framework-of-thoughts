from openai import AsyncOpenAI, OpenAIError
import backoff
from httpx import AsyncClient
from dotenv import load_dotenv
from os import getenv
import tiktoken

from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.helpers.language_model_config import Config, LLMResponseType
from llm_graph_optimizer.language_models.helpers.last_request_timer import TimingAsyncHTTPTransport
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.language_models.cache.cache import CacheContainer

class OpenAIChatWithLogprobs(AbstractLanguageModel):
    """
    Implementation of AbstractLanguageModel using OpenAI's ChatCompletion API returning tokens and logprobs.
    """

    def __init__(self, api_key=None, model: str = "gpt-4", request_price_per_token: float = 0.03, response_price_per_token: float = 0.06, config: Config = Config(), execution_cost: float = 1, cache: CacheContainer = None, openai_rate_limiter: OpenAIRateLimiter = None):
        """
        Initialize the OpenAIChat model.

        :param api_key: OpenAI API key.
        :param model: The OpenAI model to use (e.g., "gpt-4").
        :param request_price_per_token: Price per token for requests.
        :param response_price_per_token: Price per token for responses.
        :param config: Configuration for the language model.
        :param execution_cost: Cost of executing the language model.
        :param cache: Cache container.
        :param openai_rate_limiter: OpenAI rate limiter.
        """
        super().__init__(config, LLMResponseType.TOKENS_AND_LOGPROBS, execution_cost, cache)
        load_dotenv()
        api_key = api_key or getenv("OPENAI_API_KEY")
        print(api_key)
        if not api_key:
            raise ValueError("API key is not set. Please provide it or set the 'OPENAI_API_KEY' environment variable.")
        self.model = model
        self.request_price_per_token = request_price_per_token
        self.response_price_per_token = response_price_per_token

        transport = TimingAsyncHTTPTransport()
        http_client = AsyncClient(transport=transport)
        self._client = AsyncOpenAI(api_key=api_key, http_client=http_client)
        self._openai_rate_limiter = openai_rate_limiter
    @property
    def additional_cache_identifiers(self) -> dict[str, object]:
        """
        Additional identifiers for the cache.
        """
        return {
            "model": self.model,
            "request_price_per_token": self.request_price_per_token,
            "response_price_per_token": self.response_price_per_token
        }
    
    async def _raw_query(self, prompt: str) -> tuple[object, Measurement]:
        messages = [{"role": "user", "content": prompt}]
        return await self._raw_chat_query(messages)

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=120, max_tries=6)
    async def _raw_chat_query(self, prompt: list[dict[str, str]]) -> tuple[list[str, float], Measurement | None]:
        """
        Query the OpenAI Legacy Completions API and return metadata.

        :param prompt: The input prompt for the model.
        """

        if self._openai_rate_limiter:
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")  # fallback since tiktoken is not up-to-date (It is the same tokenizer for newer models)
            prompt_tokens = len(enc.encode(str(prompt)))
            estimated_response_tokens = self._config.max_tokens or 1000
            await self._openai_rate_limiter.acquire(prompt_tokens + estimated_response_tokens)
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=prompt,
                logprobs=True,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                stop=self._config.stop
            )
        finally:
            if self._openai_rate_limiter:
                headers = (
                    self._client._client._transport.last_response.headers
                )
                await self._openai_rate_limiter.sync_from_headers(headers)

        if self._openai_rate_limiter:
            delta = (prompt_tokens + estimated_response_tokens) - response.usage.total_tokens
            await self._openai_rate_limiter.adjust_tokens(delta)

        duration = self._client._client._transport.last_duration

        output_with_logprobs = [(content.token, content.logprob) for content in response.choices[0].logprobs.content]

        measurement = Measurement(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            request_cost=response.usage.prompt_tokens * self.request_price_per_token,
            response_cost=response.usage.completion_tokens * self.response_price_per_token,
            total_cost=response.usage.prompt_tokens * self.request_price_per_token + response.usage.completion_tokens * self.response_price_per_token,
            execution_cost=self._execution_cost,
            execution_duration=duration
        )
        return output_with_logprobs, measurement

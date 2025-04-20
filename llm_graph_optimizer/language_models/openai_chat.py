from httpx import AsyncClient
from openai import AsyncOpenAI
import os

from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.helpers.language_model_config import Config, LLMResponseType
from llm_graph_optimizer.language_models.helpers.last_request_timer import TimingAsyncHTTPTransport
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.language_models.cache.cache import CacheContainer

class OpenAIChat(AbstractLanguageModel):
    """
    Implementation of AbstractLanguageModel using OpenAI's ChatCompletion API.
    """

    def __init__(self, api_key=None, model: str = "gpt-4", request_price_per_token: float = 0.03, response_price_per_token: float = 0.06, config: Config = Config(), execution_cost: float = 1, cache: CacheContainer = None):
        """
        Initialize the OpenAIChat model.

        :param api_key: OpenAI API key.
        :param model: The OpenAI model to use (e.g., "gpt-4").
        :param request_price_per_token: Price per token for requests.
        :param response_price_per_token: Price per token for responses.
        """
        super().__init__(config, LLMResponseType.TEXT, execution_cost, cache)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is not set. Please provide it or set the 'OPENAI_API_KEY' environment variable.")
        self.model = model
        self.request_price_per_token = request_price_per_token
        self.response_price_per_token = response_price_per_token

        transport = TimingAsyncHTTPTransport()
        http_client = AsyncClient(transport=transport)
        self.client = AsyncOpenAI(api_key=api_key, http_client=http_client)

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

    async def _raw_query(self, prompt: str) -> tuple[str, Measurement | None]:
        """
        Query the OpenAI ChatCompletion API and return metadata.

        :param prompt: The input prompt for the model.
        :return: The response and the measurement.
        """
        # Prepare the messages for the ChatCompletion API
        messages = [{"role": "user", "content": prompt}]

        # Call the OpenAI ChatCompletion API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            stop=self._config.stop
        )
        duration = self.client._client._transport.last_duration

        # Extract token usage from the response
        measurement = Measurement(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            request_cost=response.usage.prompt_tokens * self.request_price_per_token,
            response_cost=response.usage.completion_tokens * self.response_price_per_token,
            total_cost=response.usage.prompt_tokens * self.request_price_per_token + response.usage.completion_tokens * self.response_price_per_token,
            execution_duration=duration,
            execution_cost=self._execution_cost
        )

        return response.choices[0].message.content, measurement

if __name__ == "__main__":
    import asyncio
    openai_chat = OpenAIChat(model="gpt-4o", config=Config(temperature=1.0, max_tokens=256))
    print(asyncio.run(openai_chat.query_with_logprobs("Hello, world!")))
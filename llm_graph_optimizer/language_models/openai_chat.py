from openai import AsyncOpenAI, OpenAIError
import backoff
import os

from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.query_metadata import QueryMetadata


class OpenAIChat(AbstractLanguageModel):
    """
    Implementation of AbstractLanguageModel using OpenAI's ChatCompletion API.
    """

    def __init__(self, api_key=None, model: str = "gpt-4", request_price_per_token: float = 0.03, response_price_per_token: float = 0.06, config: Config = Config()):
        """
        Initialize the OpenAIChat model.

        :param api_key: OpenAI API key.
        :param model: The OpenAI model to use (e.g., "gpt-4").
        :param request_price_per_token: Price per token for requests.
        :param response_price_per_token: Price per token for responses.
        """
        super().__init__(config)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is not set. Please provide it or set the 'OPENAI_API_KEY' environment variable.")
        self.model = model
        self.request_price_per_token = request_price_per_token
        self.response_price_per_token = response_price_per_token

        # Set the OpenAI API key
        self.client = AsyncOpenAI(api_key=api_key)

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    async def _raw_query(self, prompt: str) -> tuple[str, QueryMetadata]:
        """
        Query the OpenAI ChatCompletion API and return metadata.

        :param prompt: The input prompt for the model.
        :return: QueryMetadata containing token counts and pricing information.
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

        # Extract token usage from the response
        usage = response.usage
        request_tokens = usage.prompt_tokens
        response_tokens = usage.completion_tokens

        metadata = QueryMetadata(
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            request_price_per_token=self.request_price_per_token,
            response_price_per_token=self.response_price_per_token
        )

        # Calculate pricing
        return response.choices[0].message.content, metadata

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    async def _raw_query_with_logprobs(self, prompt: str) -> tuple[list[str, float], QueryMetadata]:
        """
        Query the OpenAI Legacy Completions API and return metadata.

        :param prompt: The input prompt for the model.
        """
        response = await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            logprobs=1,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            stop=self._config.stop
        )

        output_with_logprobs = list(zip(response.choices[0].logprobs.tokens, response.choices[0].logprobs.token_logprobs))
        return output_with_logprobs, QueryMetadata(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            request_price_per_token=self.request_price_per_token,
            response_price_per_token=self.response_price_per_token
        )


if __name__ == "__main__":
    import asyncio
    openai_chat = OpenAIChat(model="gpt-3.5-turbo-instruct", config=Config(temperature=1.0, max_tokens=256))
    print(asyncio.run(openai_chat.query_with_logprobs("Hello, world!")))
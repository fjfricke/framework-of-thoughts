from dataclasses import dataclass, field


@dataclass
class QueryMetadata:
    """
    Represents metadata for a language model query, including token counts and prices.
    """
    request_tokens: int
    response_tokens: int
    request_price_per_token: float = field(default=0.0)
    response_price_per_token: float = field(default=0.0)
    request_price: float = field(default=None)  # Optional: Directly set request price
    response_price: float = field(default=None)  # Optional: Directly set response price

    @property
    def calculated_request_price(self) -> float:
        """
        Calculate the price for the request tokens based on price per token.
        """
        return self.request_tokens * self.request_price_per_token

    @property
    def calculated_response_price(self) -> float:
        """
        Calculate the price for the response tokens based on price per token.
        """
        return self.response_tokens * self.response_price_per_token

    @property
    def effective_request_price(self) -> float:
        """
        Return the directly set request price if available, otherwise calculate it.
        """
        if self.request_price is not None:
            return self.request_price
        return self.request_tokens * self.request_price_per_token

    @property
    def effective_response_price(self) -> float:
        """
        Return the directly set response price if available, otherwise calculate it.
        """
        if self.response_price is not None:
            return self.response_price
        return self.response_tokens * self.response_price_per_token

    @property
    def total_tokens(self) -> int:
        """
        Calculate the total number of tokens (request + response).
        """
        return self.request_tokens + self.response_tokens

    @property
    def total_price(self) -> float:
        """
        Calculate the total price (request + response).
        """
        return self.effective_request_price + self.effective_response_price 
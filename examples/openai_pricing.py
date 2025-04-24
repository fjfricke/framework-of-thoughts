# prices of 22.04.2025

OPENAI_PRICING = {
    "gpt-4.1": {
        "request_price_per_token": 2.0 / 1_000_000,
        "cached_input_price_per_token": 0.5 / 1_000_000,
        "response_price_per_token": 8.0 / 1_000_000,
    },
    "gpt-4.1-mini": {
        "request_price_per_token": 0.4 / 1_000_000,
        "cached_input_price_per_token": 0.1 / 1_000_000,
        "response_price_per_token": 1.6 / 1_000_000,
    },
    "gpt-4.1-nano": {
        "request_price_per_token": 0.1 / 1_000_000,
        "cached_input_price_per_token": 0.025 / 1_000_000,
        "response_price_per_token": 0.4 / 1_000_000,
    },
    "gpt-4.5-preview": {
        "request_price_per_token": 75.0 / 1_000_000,
        "cached_input_price_per_token": 37.5 / 1_000_000,
        "response_price_per_token": 150.0 / 1_000_000,
    },
    "gpt-4o": {
        "request_price_per_token": 2.5 / 1_000_000,
        "cached_input_price_per_token": 1.25 / 1_000_000,
        "response_price_per_token": 10.0 / 1_000_000,
    },
    "gpt-4o-audio-preview": {
        "request_price_per_token": 2.5 / 1_000_000,
        "response_price_per_token": 10.0 / 1_000_000,
    },
    "gpt-4o-realtime-preview": {
        "request_price_per_token": 5.0 / 1_000_000,
        "cached_input_price_per_token": 2.5 / 1_000_000,
        "response_price_per_token": 20.0 / 1_000_000,
    },
    "gpt-4o-mini": {
        "request_price_per_token": 0.15 / 1_000_000,
        "cached_input_price_per_token": 0.075 / 1_000_000,
        "response_price_per_token": 0.6 / 1_000_000,
    },
    "gpt-4o-mini-audio-preview": {
        "request_price_per_token": 0.15 / 1_000_000,
        "response_price_per_token": 0.6 / 1_000_000,
    },
    "gpt-4o-mini-realtime-preview": {
        "request_price_per_token": 0.6 / 1_000_000,
        "cached_input_price_per_token": 0.3 / 1_000_000,
        "response_price_per_token": 2.4 / 1_000_000,
    },
    "o1": {
        "request_price_per_token": 15.0 / 1_000_000,
        "cached_input_price_per_token": 7.5 / 1_000_000,
        "response_price_per_token": 60.0 / 1_000_000,
    },
    "o1-pro": {
        "request_price_per_token": 150.0 / 1_000_000,
        "response_price_per_token": 600.0 / 1_000_000,
    },
    "o3": {
        "request_price_per_token": 10.0 / 1_000_000,
        "cached_input_price_per_token": 2.5 / 1_000_000,
        "response_price_per_token": 40.0 / 1_000_000,
    },
    "o4-mini": {
        "request_price_per_token": 1.1 / 1_000_000,
        "cached_input_price_per_token": 0.275 / 1_000_000,
        "response_price_per_token": 4.4 / 1_000_000,
    },
    "o3-mini": {
        "request_price_per_token": 1.1 / 1_000_000,
        "cached_input_price_per_token": 0.55 / 1_000_000,
        "response_price_per_token": 4.4 / 1_000_000,
    },
    "o1-mini": {
        "request_price_per_token": 1.1 / 1_000_000,
        "cached_input_price_per_token": 0.55 / 1_000_000,
        "response_price_per_token": 4.4 / 1_000_000,
    },
    "gpt-4o-mini-search-preview": {
        "request_price_per_token": 0.15 / 1_000_000,
        "response_price_per_token": 0.6 / 1_000_000,
    },
    "gpt-4o-search-preview": {
        "request_price_per_token": 2.5 / 1_000_000,
        "response_price_per_token": 10.0 / 1_000_000,
    },
    "computer-use-preview": {
        "request_price_per_token": 3.0 / 1_000_000,
        "response_price_per_token": 12.0 / 1_000_000,
    },
}

# legacy models

OPENAI_PRICING.update({
    "chatgpt-4o-latest": {
        "request_price_per_token": 5.0 / 1_000_000,
        "response_price_per_token": 15.0 / 1_000_000,
    },
    "gpt-4-turbo": {
        "request_price_per_token": 10.0 / 1_000_000,
        "response_price_per_token": 30.0 / 1_000_000,
    },
    "gpt-4": {
        "request_price_per_token": 30.0 / 1_000_000,
        "response_price_per_token": 60.0 / 1_000_000,
    },
    "gpt-4-32k": {
        "request_price_per_token": 60.0 / 1_000_000,
        "response_price_per_token": 120.0 / 1_000_000,
    },
    "gpt-3.5-turbo": {
        "request_price_per_token": 0.5 / 1_000_000,
        "response_price_per_token": 1.5 / 1_000_000,
    },
    "gpt-3.5-turbo-instruct": {
        "request_price_per_token": 1.5 / 1_000_000,
        "response_price_per_token": 2.0 / 1_000_000,
    },
    "gpt-3.5-turbo-16k-0613": {
        "request_price_per_token": 3.0 / 1_000_000,
        "response_price_per_token": 4.0 / 1_000_000,
    },
    "davinci-002": {
        "request_price_per_token": 2.0 / 1_000_000,
        "response_price_per_token": 2.0 / 1_000_000,
    },
    "babbage-002": {
        "request_price_per_token": 0.4 / 1_000_000,
        "response_price_per_token": 0.4 / 1_000_000,
    },
})
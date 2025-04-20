import httpx
import time

class TimingAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_duration = None

    async def handle_async_request(self, request):
        start = time.perf_counter()
        response = await super().handle_async_request(request)
        end = time.perf_counter()
        self.last_duration = end - start
        return response
"""Select translations per request without buffering responses or changing stored data."""

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from chap_core.i18n import current_translations, load_translations, select_language


class TranslationMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        translations = load_translations()
        language = select_language(Headers(scope=scope).get("accept-language", ""), translations)
        token = current_translations.set(translations[language])

        async def send_response(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("Content-Language", language)
                headers.add_vary_header("Accept-Language")
            await send(message)

        try:
            await self.app(scope, receive, send_response)
        finally:
            current_translations.reset(token)

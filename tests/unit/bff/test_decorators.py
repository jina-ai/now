import asyncio
import time

from now.bff.decorators import api_method, async_timed, timed


def test_timed():
    @timed
    def monty():
        """Monty Python!"""
        time.sleep(0.1)

    monty()


def test_async_timed():
    @async_timed
    async def monty():
        """Monty Python!"""
        await asyncio.sleep(0.1)

    asyncio.run(monty())


def test_api_method():
    @api_method
    def monty():
        """Monty Python!"""
        time.sleep(0.1)

    monty()

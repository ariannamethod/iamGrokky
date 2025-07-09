import asyncio
import inspect

# Simple hook to run async test functions using asyncio.run

def pytest_pyfunc_call(pyfuncitem):
    test_func = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_func):
        asyncio.run(test_func(**pyfuncitem.funcargs))
        return True
    return None

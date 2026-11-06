import os


def calculator_mode():
    if os.getenv("NEUROMEM_ENABLE_WOLFRAM") == "1":
        if not os.getenv("WOLFRAMALPHA_APP_ID"):
            raise ValueError("Live Calculate requires WOLFRAMALPHA_APP_ID.")
        return "live_wolfram"
    return "offline"


def WolframAlphaCalculator(query):
    if calculator_mode() == "offline":
        # Disabled calculator calls return an error without contacting a service.
        raise Exception("Invalid input query for Calculator. Please check the input query or use other functions to do the computation.")
    import wolframalpha

    result = wolframalpha.Client(os.environ["WOLFRAMALPHA_APP_ID"]).query(query)
    return next(result.results).text

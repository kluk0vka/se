from backend.agent import agent

def test_sentiment():
    result = agent.run("я очень доволен этим фильмом!")
    assert "positive" in result.lower() or "положительн" in result.lower()

def test_translate():
    result = agent.run("привет")
    assert ("hi" in result.lower() or "hello" in result.lower() or
            "привет" in result.lower())

def test_summarize():
    result = agent.run("Сегодня я пошёл в магазин и купил молоко, хлеб и сыр.")
    assert any(w in result.lower() for w in ("магазин", "продукты", "молоко", "shop", "bread"))
from backend.tools import sentiment, translate, summarize

def test_sentiment():
    assert sentiment("я рад") == "POSITIVE"

def test_translate():
    assert "cat" in translate("кот").lower()

def test_summarize():
    assert any(w in summarize("я пошёл в магазин за хлебом").lower() for w in ("магазин", "хлеб"))
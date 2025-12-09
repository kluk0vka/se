from backend.model import get_sentiment
from backend.tools import translate, summarize

def test_sentiment():
    assert get_sentiment("я рад") == "POSITIVE"

def test_translate():
    ans = translate("привет")
    assert "hi" in ans.lower() or "hello" in ans.lower()

def test_summarize():
    out = summarize("Сегодня я пошёл в магазин за хлебом.")
    assert any(w in out.lower() for w in ("магазин", "продукты", "молоко"))
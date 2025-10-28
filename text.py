from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# модель: blanchefort/rubert-base-cased-sentiment (русскоязычная, обучена на отзывах)
tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment")

texts = [
    "этот мем такой угарный",
    "вчерашний стрим был супер вайбовым",
    "нормуль кинчик",
    "ну это полный минус вайб",
    "её новый лук такой слэй, теперь хочу такой же",
    "тот чел с тиктока полный кринж",
]

# (0 — негативный, 1 — нейтральный, 2 — позитивный)
true_labels = [2, 2, 1, 0, 2, 0]


classes = ["негативный", "нейтральный", "позитивный"]
pred_labels = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1).squeeze().tolist()
    pred_label = torch.argmax(torch.tensor(probs)).item()
    pred_labels.append(pred_label)

    print(f"текст: {text}")
    for cls, prob in zip(classes, probs):
        print(f"вероятность, что текст {cls} — {prob * 100:.0f}%")
    print(f"итоговая классификация: {classes[pred_label]}")
    print("-" * 40)

acc = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
cm = confusion_matrix(true_labels, pred_labels)

print("\nметрики качества модели:")
print(f"Accuracy:  {acc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}")

print("\nматрица ошибок (confusion matrix):")
print(cm)
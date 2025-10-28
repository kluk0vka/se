import torch
from torchvision.io import read_video
from torchvision.transforms import Resize
from torchvision.models.video import mc3_18
import torch.nn.functional as F

video_path = "vidos.mp4"
video, _, info = read_video(video_path, pts_unit='sec')

video = video[::2]

video = video.permute(0,3,1,2).float() / 255.0

resize = Resize((112,112))
video = torch.stack([resize(frame) for frame in video])

video = video.permute(1,0,2,3).unsqueeze(0)

model = mc3_18(weights="MC3_18_Weights.KINETICS400_V1")
model.eval()

with torch.no_grad():
    outputs = model(video)
    probs = F.softmax(outputs, dim=1)

with open("kinetics_classnames.txt") as f:
    categories = [line.strip() for line in f.readlines()]

top_prob, top_catid = torch.topk(probs, 1)
predicted_action = categories[top_catid[0][0].item()]
predicted_prob = top_prob[0][0].item()

print(f"предсказанное действие (топ-1): {predicted_action}, вероятность: {predicted_prob:.2f}")

top_prob5, top_catid5 = torch.topk(probs, 5, dim=1)
top_prob5 = top_prob5[0]
top_catid5 = top_catid5[0]

print("\nтоп-5 возможных действий:")
for i in range(5):
    print(f"{categories[top_catid5[i].item()]}: {top_prob5[i].item():.2f}")

with open("video_action_results_pytorch.txt", "w") as f:
    f.write(f"предсказанное действие (топ-1): {predicted_action}, вероятность: {predicted_prob:.2f}\n")
    f.write("\nтоп-5 возможных действий:\n")
    for i in range(5):
        f.write(f"{categories[top_catid5[i].item()]}: {top_prob5[i].item():.2f}\n")



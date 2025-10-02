import os
import requests
from io import BytesIO
from flask import Flask, request, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# flask initialize
app = Flask(__name__)

# model loading
# モデル全体をロード
model = torch.load("model.pth", map_location=torch.device("cpu"))
model.eval()


# image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def predict_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probabilities = F.softmax(output, dim=1)[0]
        predicted = torch.argmax(probabilities).item()
        class_names = ["Cat", "Dog"]
        return class_names[predicted], probabilities.tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            try:
                prediction, probs = predict_from_url(url)
                return render_template("index.html",
                                       prediction=prediction,
                                       probs=probs,
                                       url=url)
            except Exception as e:
                return render_template("index.html", prediction=f"エラー: {str(e)}")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

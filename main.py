from flask import Flask, request, render_template, redirect
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__)

# Load the model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 200)
model.load_state_dict(torch.load('tinyimagenet_resnet18.pth', map_location='cpu'))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image = Image.open(file.stream)
            image = transform(image).unsqueeze(0)  # Add batch dimension

            # Make a prediction
            with torch.no_grad():
                output = model(image)
                probabilities = F.softmax(output, dim=1)  # Get probabilities
                confidence, predicted_class = probabilities.max(1)
                predicted_class = predicted_class.item()
                confidence = confidence.item() * 100  # Convert to percentage

            # Render results with prediction and confidence
            return render_template('index.html', prediction=predicted_class, confidence=confidence)

    return render_template('index.html', prediction=None, confidence=None)

if __name__ == '__main__':
    app.run(debug=True)

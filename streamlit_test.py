import torch
import streamlit as st
from models.EfficientnetB0 import EfficientnetB0
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F

@st.cache_resource

def load_model():
    ckpt_path = "/Users/fredvaartnou/VSCODE/PestSpace/Checkpoints/best.ckpt" 
    model = EfficientnetB0.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])


st.title("Plant Disease Detection")

uploaded_file = st.file_uploader(
    "Upload image", type=["jpg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    classes = ["Downy mildew", "Chocolate spot", "Soy rust"]

    results = list(zip(classes, probs[0].tolist()))
    results = sorted(results, key=lambda x: x[1], reverse=True)

    for rank, (cls, p) in enumerate(results, 1):
        st.write(f"{rank}. {cls}: {p*100:.1f}%")


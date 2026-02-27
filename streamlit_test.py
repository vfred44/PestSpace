import torch
import streamlit as st
from models.EfficientnetB0 import EfficientnetB0
from torchvision import transforms
from PIL import Image
import pillow_heif
import torch
import torch.nn.functional as F
import os

Image.MAX_IMAGE_PIXELS = None
pillow_heif.register_heif_opener()

# Example image paths:

Downy_mildew_examples = '/Users/fredvaartnou/Desktop/Downy_mildew_examples/'
Chocolate_spot_examples = '/Users/fredvaartnou/Desktop/Chocolate_spot_examples'
Soy_rust_examples = '/Users/fredvaartnou/Desktop/Soy_rust_examples'

causing_taxa = {"Downy mildew": {"Peronospora viciae": "https://elurikkus.ee/app/taxonomy/taxon/192883"},
                "Chocolate spot": {"Botrytis fabae": "https://elurikkus.ee/app/taxonomy/taxon/135558",
                                   "Botrytis cinerea": "https://elurikkus.ee/app/taxonomy/taxon/135549"},
                "Soy rust": {"Phakopsora pachyrhizi": "https://elurikkus.ee/app/taxonomy/taxon/164595",
                             "Phakopsora meibomiae": "https://elurikkus.ee/app/taxonomy/taxon/164591"}
                }



@st.cache_resource

def load_model():
    ckpt_path = "/Users/fredvaartnou/VSCODE/PestSpace/Checkpoints/19.01.2026/epoch=24-val_loss=0.0471.ckpt" 
    model = EfficientnetB0.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

model = load_model()
print(model)

transform = transforms.Compose([
    transforms.Resize(360),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
])

classes = ["Downy mildew", "Chocolate spot", "Soy rust"]

st.title("Plant Disease Detection")

uploaded_files = st.file_uploader(
    "Upload up to three images", 
    type=["jpg", "png", "HEIC"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 3:
        st.error("Too many images")
    
    else:
        images = []
        tensors = []

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
            tensors.append(transform(image))

        batch = torch.stack(tensors)

    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
    
    #Average probabilities across images
    avg_probs = probs.mean(dim=0)
   
    results = list(zip(classes, avg_probs.tolist()))
    results.sort (key=lambda x: x[1], reverse=True)


    for rank, (cls, p) in enumerate(results, 1):
        if p > 0.01:
            st.subheader(f"{rank}. {cls}: {p*100:.1f} %")
            
    st.markdown("<br>", unsafe_allow_html=True)

    # Example pictures

    example_dirs = {
        "Downy mildew": Downy_mildew_examples,
        "Chocolate spot": Chocolate_spot_examples,
        "Soy rust": Soy_rust_examples
    }

   
    for rank, (cls, p) in enumerate(results, 1):  
        if rank <= 3 and p >= 0.1:
            st.markdown(f"### {cls}")
            if cls in causing_taxa:
                taxa_dict = causing_taxa[cls]
                links = " - ".join(
                    f"[{taxa}]({url})"
                    for taxa, url in taxa_dict.items())
                st.markdown(f"**Causing taxa:** {links}")
            
            st.markdown("**Example images:**")
            folder = example_dirs.get(cls)
            example_images = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(".jpg")
            ]

            cols = st.columns(len(example_images))

            for col, img_path in zip(cols, example_images):
                image = Image.open(img_path)
                col.image(image, width="stretch")
        
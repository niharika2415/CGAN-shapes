import streamlit as st
import torch
from torchvision.utils import save_image
from PIL import Image
import os

# Config 
IMG_SIZE = 64
Z_DIM = 100
LABELS = ["circle", "square", "triangle", "rectangle"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "/content/drive/MyDrive/06-CGAN-shapes/cgan_checkpoints"
G_CHECKPOINT = os.path.join(SAVE_DIR, "checkpoint_epoch_040.pth")


from cgan_models import Generator  

# Load Generator
@st.cache_resource
def load_generator():
    G = Generator().to(DEVICE)
    checkpoint = torch.load(G_CHECKPOINT, map_location=DEVICE)
    G.load_state_dict(checkpoint['G_state'])
    G.eval()
    return G

G = load_generator()

# GUI 
st.title("CGAN Shape Generator")
st.write("Generate shapes using a Conditional GAN.")

shape = st.selectbox("Select Shape", LABELS)
num_samples = st.slider("Number of Samples", min_value=1, max_value=16, value=8)

if st.button("Generate"):
    idx = LABELS.index(shape)
    z = torch.randn(num_samples, Z_DIM, device=DEVICE)
    labels = torch.tensor([idx]*num_samples, dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        gen_imgs = G(z, labels).cpu()
    
    # save temporary image grid
    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/{shape}_samples.png"
    save_image(gen_imgs, out_path, nrow=num_samples, normalize=False)
    
    # display
    st.image(out_path, caption=f"{shape.capitalize()} Samples")

import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

# Config
IMG_SIZE = 64
Z_DIM = 100
LABELS = ["circle", "square", "triangle", "rectangle"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "/content/drive/MyDrive/06-CGAN-shapes/cgan_checkpoints"
G_CHECKPOINT = os.path.join(SAVE_DIR, "checkpoint_epoch_040.pth")

# Helper: View for Generator
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, label_emb_dim=50, num_classes=len(LABELS), out_channels=1):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        input_dim = z_dim + label_emb_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512*4*4),
            nn.BatchNorm1d(512*4*4),
            nn.ReLU(True),
            View((-1, 512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1, bias=False),  # 64x64
            nn.Sigmoid()
        )

    def forward(self, z, labels):
        le = self.label_emb(labels)
        x = torch.cat([z, le], dim=1)
        return self.net(x)

# Load Generator
@st.cache_resource
def load_generator():
    G = Generator().to(DEVICE)
    checkpoint = torch.load(G_CHECKPOINT, map_location=DEVICE)
    G.load_state_dict(checkpoint['G_state'])
    G.eval()
    return G

G = load_generator()

# Streamlit GUI
st.title("CGAN Shape Generator")
st.write("Generate simple shapes using a trained Conditional GAN.")

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

    st.image(out_path, caption=f"{shape.capitalize()} Samples")

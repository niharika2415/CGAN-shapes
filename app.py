import os
import torch
import streamlit as st
from torchvision.utils import save_image
from cgan_models import Generator  # make sure this file is in your repo
from typing import List

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Checkpoint
G_CHECKPOINT = os.path.join("cgan_checkpoints", "checkpoint_epoch_040.pth")
CHECKPOINT_URL = "https://drive.google.com/file/d/1l-KS27b0mmVgfGFI6XFgBJAX5BU1jQ0i/view?usp=sharing"

# Download checkpoint if not exists
if not os.path.exists(G_CHECKPOINT):
    import gdown
    os.makedirs("cgan_checkpoints", exist_ok=True)
    gdown.download(CHECKPOINT_URL, G_CHECKPOINT, quiet=False)


# Load Generator
@st.cache_resource
def load_generator():
    G = Generator().to(DEVICE)
    checkpoint = torch.load(G_CHECKPOINT, map_location=DEVICE)
    G.load_state_dict(checkpoint['G_state'])
    G.eval()
    return G

G = load_generator()


# Sampling function
def generate_samples(model: torch.nn.Module, labels: List[str], samples_per_label: int = 8):
    from cgan_train import LABELS, Z_DIM  # adjust import if needed

    label_idxs = [LABELS.index(l) for l in labels]
    z = torch.randn(len(label_idxs) * samples_per_label, Z_DIM, device=DEVICE)
    labels_tensor = torch.tensor(
        [idx for idx in label_idxs for _ in range(samples_per_label)],
        dtype=torch.long,
        device=DEVICE
    )
    with torch.no_grad():
        gen = model(z, labels_tensor).cpu()

    save_path = os.path.join(SAVE_DIR, "samples.png")
    save_image(gen, save_path, nrow=samples_per_label, normalize=False)
    return save_path

# Streamlit GUI
st.title("🖌️ CGAN Shape Generator")

# Label selection
from cgan_train import LABELS  # adjust import if needed
selected_labels = st.multiselect("Select shapes to generate:", LABELS, default=LABELS)

samples_per_label = st.slider("Samples per label:", 1, 16, 8)

if st.button("Generate"):
    if not selected_labels:
        st.warning("Select at least one shape!")
    else:
        st.info("Generating images...")
        out_file = generate_samples(G, selected_labels, samples_per_label)
        st.success("Generated samples!")
        st.image(out_file, use_column_width=True)

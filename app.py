import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
from model_architecture import CNNtoRNN, Vocabulary # Import our classes
import torchvision.models as models

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_resources():
    # 1. Load Vocab
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    # 2. Initialize Model
    model = CNNtoRNN(embed_size=256, hidden_size=256, vocab_size=len(vocab), num_layers=1).to(device)
    
    # 3. Load Weights
    # map_location=device allows running on CPU even if trained on GPU
    model.load_state_dict(torch.load('model_final_run_epoch_35.pth', map_location=device))
    model.eval()

    # 4. Load ResNet for feature extraction
    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    resnet.to(device)
    resnet.eval()
    
    return vocab, model, resnet

vocab, model, resnet_extractor = load_resources()

# --- PREDICTION FUNCTION ---
def generate_caption(image, model, vocab, resnet_extractor):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = resnet_extractor(img_tensor).reshape(1, -1)
        encoder_out = model.encoder(features)
    
    result_caption = []
    inputs = encoder_out.unsqueeze(1)
    states = None
    
    for _ in range(20):
        hiddens, states = model.decoder.lstm(inputs, states)
        output = model.decoder.linear(hiddens.squeeze(1))
        predicted = output.argmax(1)
        word = vocab.itos[predicted.item()]
        
        if word == "<EOS>": break
        if word != "<SOS>": result_caption.append(word)
        
        inputs = model.decoder.embed(predicted).unsqueeze(1)
        
    return ' '.join(result_caption)

# --- STREAMLIT UI ---
st.title("📸 Neural Storyteller")
st.write("Upload an image and my AI will try to describe it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image',  use_container_width=True)
    
    if st.button('Generate Caption'):
        with st.spinner('AI is thinking...'):
            caption = generate_caption(image, model, vocab, resnet_extractor)

            st.success(f"**AI says:** {caption}")

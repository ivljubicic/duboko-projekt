import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import io
import copy
import time

st.sidebar.subheader("Settings")

transfer_mode = st.sidebar.selectbox(
    "Style Transfer Mode",
    ["Single Style", "Dual Style"],
    help="Choose between one or two style images"
)

style_blend = 0.5
if transfer_mode == "Dual Style":
    style_blend = st.sidebar.slider("Style Blend Ratio (Style1:Style2)", 0.0, 1.0, 0.5, 0.1)

image_size = st.sidebar.slider("Output Image Size", 128, 1024, 512, 128)

num_iterations = st.sidebar.slider(
    "Number of Iterations",
    100,
    500,
    300,
    50,
    help="More iterations generally give better results but take longer"
)

st.sidebar.subheader("Style-Content Balance")
content_strength = st.sidebar.slider(
    "Content Strength",
    0.1,
    100.0,
    1.0,
    help="Higher values make the output preserve more of the content image"
)

style_strength = st.sidebar.slider(
    "Style Strength",
    0.1,
    100.0,
    1.0,
    help="Higher values make the output look more like the style image(s)"
)

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()])

def content_image_loader(image):
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def style_image_loader(image, size):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)

def load_image(image_file):
    img = Image.open(image_file)
    
    quality = 95
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=quality)
    while buffered.getbuffer().nbytes > 2 * 1024 * 1024 and quality > 10:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        buffered.seek(0)
        quality -= 5
    img = Image.open(buffered)
    
    return img

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class StyleLoss(nn.Module):
    def __init__(self, target_feature1, target_feature2=None, blend_ratio=1.0):
        super(StyleLoss, self).__init__()
        self.target1 = self.gram_matrix(target_feature1).detach()
        if target_feature2 is not None:
            self.target2 = self.gram_matrix(target_feature2).detach()
            self.combined_target = blend_ratio * self.target1 + (1 - blend_ratio) * self.target2
        else:
            self.combined_target = self.target1

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.combined_target)
        return input

    def gram_matrix(self, input):
        batch_size, f_map_num, h, w = input.size()
        features = input.view(batch_size * f_map_num, h * w)
        G = torch.mm(features, features.t())
        return G.div(batch_size * f_map_num * h * w)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                              style_img1, style_img2, content_img, blend_ratio=1.0):
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature1 = model(style_img1).detach()
            target_feature2 = model(style_img2).detach() if style_img2 is not None else None
            style_loss = StyleLoss(target_feature1, target_feature2, blend_ratio)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                      content_img, style_img1, style_img2=None, input_img=None, 
                      blend_ratio=1.0, num_steps=300,
                      style_weight=1000000, content_weight=1):
    if input_img is None:
        input_img = content_img.clone()
        
    style_weight = style_weight * 100000
    
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, 
        style_img1, style_img2, content_img, blend_ratio
    )
    
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])
    
    st.write(f"Showing progress of {num_steps} iterations:")
    progress_bar = st.progress(0)
    
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            
            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            # update progress bar
            progress_bar.progress(min(run[0] / num_steps, 1.0))
            
            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    progress_bar.empty()
    return input_img

def main():
    st.title("Neural Style Transfer")
    st.write("Upload your content and style images to create artistic compositions")

    if transfer_mode == "Single Style":
        col1, col2 = st.columns(2)
        col3 = None
    else:
        col1, col2, col3 = st.columns(3)

    content_image = None
    style_image1 = None
    style_image2 = None

    with col1:
        st.subheader("Content Image")
        content_file = st.file_uploader("Choose your target image", type=["png", "jpg", "jpeg"])
        if content_file is not None:
            content_image = load_image(content_file)
            st.image(content_image, caption="Target Image", use_container_width=True)

    with col2:
        st.subheader("Style Image 1")
        style_file1 = st.file_uploader("Choose your first style image", type=["png", "jpg", "jpeg"])
        if style_file1 is not None:
            style_image1 = load_image(style_file1)
            st.image(style_image1, caption="Style Image 1", use_container_width=True)

    if transfer_mode == "Dual Style" and col3:
        with col3:
            st.subheader("Style Image 2")
            style_file2 = st.file_uploader("Choose your second style image", type=["png", "jpg", "jpeg"])
            if style_file2 is not None:
                style_image2 = load_image(style_file2)
                st.image(style_image2, caption="Style Image 2", use_container_width=True)

    can_generate = (
        content_image is not None and 
        style_image1 is not None and 
        (transfer_mode == "Single Style" or 
         (transfer_mode == "Dual Style" and style_image2 is not None))
    )

    if can_generate:
        if st.button("Generate Styled Image"):
            with st.spinner("Doing style transfer! Be patient, this may take a while..."):
                start_time = time.time()

                content_img = content_image_loader(content_image)
                content_size = content_img.size()[2:]
                style_img1 = style_image_loader(style_image1, content_size)
                style_img2 = None
                if transfer_mode == "Dual Style":
                    style_img2 = style_image_loader(style_image2, content_size)
                
                input_image = content_img.clone()
                output_image = run_style_transfer(
                    cnn, cnn_normalization_mean, cnn_normalization_std,
                    content_img, style_img1, style_img2, input_image,
                    blend_ratio=style_blend,
                    num_steps=num_iterations,
                    style_weight=style_strength,
                    content_weight=content_strength
                )
                
                output_image = output_image.cpu().squeeze(0)
                output_image = transforms.ToPILImage()(output_image)

                st.subheader("Generated Image")

                end_time = time.time() 
                elapsed_time = end_time - start_time

                st.write(f"Transformation completed in {elapsed_time:.2f} seconds.")
                st.image(output_image, caption="Style Transfer Result", use_container_width=True)
                
                buffered = io.BytesIO()
                output_image.save(buffered, format="PNG")
                st.download_button(
                    label="Download image",
                    data=buffered.getvalue(),
                    file_name="style_transfer_result.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
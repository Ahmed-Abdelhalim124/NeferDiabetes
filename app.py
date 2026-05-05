# %%
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    import numpy as np
    from PIL import Image
    import timm
    import gradio as gr
    import matplotlib.pyplot as plt

    
    class DRClassifier(nn.Module):
        def __init__(self, num_classes=5, model_name='tf_efficientnet_b3_ns'):
            super(DRClassifier, self).__init__()
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
            num_features = self.backbone.num_features
            self.classifier = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)

    
    def load_checkpoint_safely(checkpoint_path, model, device):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            loaded_components = {'model': False}
            
            if 'model_state_dict' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    loaded_components['model'] = True
                except Exception as e:
                    print(f"Failed to load model state: {e}")
            
            return checkpoint, loaded_components
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None, {'model': False}

    
    def predict_single_image(image, model, device, class_names):
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = image.convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
                predicted_idx = torch.argmax(outputs, dim=1).item()
            
            
            result = {
                "Predicted Class": class_names[predicted_idx],
                "Confidence Scores": {class_names[i]: f"{prob*100:.2f}%" for i, prob in enumerate(probabilities)}
            }
            
            
            plt.figure(figsize=(8, 4))
            plt.bar(class_names, probabilities)
            plt.xlabel("Classes")
            plt.ylabel("Probability")
            plt.title("Prediction Confidence Scores")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return result, plt.gcf()
        except Exception as e:
            return f"Error processing image: {e}", None

    
    def main():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        
        class_names = ['Healthy', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferate DR']

        
        model = DRClassifier(num_classes=5, model_name='tf_efficientnet_b3_ns').to(device)
        
        
        checkpoint_path = r"D:\Neferx\Neferdiabetes\Neferdiabetes\dr_model.pth"
        checkpoint, loaded_components = load_checkpoint_safely(checkpoint_path, model, device)
        if not loaded_components['model']:
            return "Error: Failed to load model weights.", None

        
        def gradio_predict(image):
            return predict_single_image(image, model, device, class_names)

        with gr.Blocks() as interface:
            gr.Markdown("# Diabetic Retinopathy Prediction Demo")
            gr.Markdown("Upload a retinal image to predict its diabetic retinopathy class.")
            
            image_input = gr.Image(type="pil", label="Upload Retinal Image")
            predict_button = gr.Button("Predict")
            output_json = gr.JSON(label="Prediction Result")
            output_plot = gr.Plot(label="Confidence Scores")
            
            predict_button.click(
                fn=gradio_predict,
                inputs=image_input,
                outputs=[output_json, output_plot]
            )
        
        interface.launch()

    if __name__ == "__main__":
        main()



# %%

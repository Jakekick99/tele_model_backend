import torch
import torchvision.transforms as transforms
from django.http import JsonResponse

# Load the trained model
model = torch.load('path_to_your_model.pth')
model.eval()

# Define the transformation to apply to input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(request):
    # Retrieve the input data from the request
    image_file = request.FILES.get('image')

    if not image_file:
        return JsonResponse({'error': 'No image file provided.'})

    # Preprocess the input image
    image = transform(image_file)
    image = image.unsqueeze(0)

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = str(predicted_idx.item())

    # Create the prediction response
    prediction = {
        'predicted_label': predicted_label,
    }

    return JsonResponse(prediction)
# import torch
# import torchvision.transforms as transforms
from django.http import JsonResponse
from PIL import Image
import numpy as np
from skimage import transform
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt

from huggingface_hub import from_pretrained_keras
# Load Model
retrieved_model = from_pretrained_keras("Madhu45/Teledermatology_model")
print("Model Loaded.")
CLASS_NAMES = ['1. Enfeksiyonel', '2. Ekzama', '3. Akne', '4. Pigment', '5. Benign', '6. Malign']

def load_image(image_obj):
    image = Image.open(image_obj)
    image = np.array(image).astype('float32')
    image = transform.resize(image, (32, 32, 3))
    image = np.expand_dims(image, axis=0)
    return image
# Define the transformation to apply to input data
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# # ])

@csrf_exempt
def predict(request):
    # Retrieve the input data from the request
    image_file = request.FILES.get('image')

    if not image_file:
        return JsonResponse({'error': 'No image file provided.'}, status=500)

    try:
        # Load Image
        input_image = load_image(image_file)
        print("Image Loaded.")

        # Getting Predictions and
        prediction = retrieved_model.predict(input_image)
        score = tf.nn.softmax(prediction[0])
        predicted_class = CLASS_NAMES[np.argmax(score)]
        prediction_confidence = 100 * np.max(score)
        print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(predicted_class,
                                                                                              prediction_confidence))

        return JsonResponse({
            "msg": "Successful Inference",
            "predicted_class": predicted_class,
            "confidence": prediction_confidence
            },status=200)

    except Exception as e:
        print(e)
        return JsonResponse({"msg": "Inference Failed!!"}, status=500)



#     # Preprocess the input image
#     image = transform(image_file)
#     image = image.unsqueeze(0)
#
#     # Perform the prediction
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted_idx = torch.max(outputs, 1)
#         predicted_label = str(predicted_idx.item())
#
#     # Create the prediction response
#     prediction = {
#         'predicted_label': predicted_label,
#     }

    return JsonResponse(prediction)
import boto3
import json
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

def create_custom_inference_script():
    script = """
import json
import torch
from PIL import Image
import requests
from io import BytesIO
import base64
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def model_fn(model_dir):
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir)
    return model, processor

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        image_data = input_data['inputs']['image']
        text = input_data['inputs']['text']
        
        # Decode base64 image
        image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
        
        return {'image': image, 'text': text}
    else:
        raise ValueError('Unsupported content type: ' + request_content_type)

def predict_fn(input_data, model_and_processor):
    model, processor = model_and_processor
    
    inputs = processor(images=input_data['image'], text=input_data['text'], return_tensors="pt")
    outputs = model(**inputs)
    
    # Process the outputs as needed
    target_sizes = torch.tensor([input_data['image'].size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    
    return results

def output_fn(prediction_output, accept):
    return json.dumps(prediction_output)
"""
    with open('inference.py', 'w') as f:
        f.write(script)
    print("Custom inference script created.")

def deploy_model():
    role = sagemaker.get_execution_role()
    
    huggingface_model = HuggingFaceModel(
        model_data="s3://your-model-bucket/model.tar.gz",  # Update this with your model's S3 path
        role=role,
        transformers_version="4.37.0",
        pytorch_version="2.1.0",
        py_version='py310',
        entry_point='inference.py'
    )
    
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge',
        endpoint_name='grounding-dino-endpoint'
    )
    
    print(f"Model deployed. Endpoint name: {predictor.endpoint_name}")
    return predictor.endpoint_name

def invoke_endpoint(endpoint_name, image_data, text):
    client = boto3.client('sagemaker-runtime')
    
    payload = {
        "inputs": {
            "image": image_data,
            "text": text
        }
    }
    
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    result = json.loads(response['Body'].read().decode())
    return result

# Main execution
if __name__ == "__main__":
    create_custom_inference_script()
    endpoint_name = deploy_model()
    
    # Example usage
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    text = "a cat. a remote control."
    
    # Download and encode image
    response = requests.get(image_url)
    image_data = base64.b64encode(response.content).decode('utf-8')
    
    try:
        result = invoke_endpoint(endpoint_name, image_data, text)
        print("Inference result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error invoking endpoint: {str(e)}")
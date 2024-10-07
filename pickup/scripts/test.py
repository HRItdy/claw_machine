import requests

# Replace with the actual ngrok URL
url = "https://08de-34-169-37-63.ngrok-free.app/infer"  # Update the endpoint to '/infer'

file_path = "/home/lab_cheem/Pictures/screenshot_2.png"  # Replace with your local image path
text_prompt = "a white ball"  # Replace with your desired text prompt

files = {'image': open(file_path, 'rb')}
data = {'text_prompt': text_prompt}

try:
    # Set a timeout of 10 seconds
    response = requests.post(url, files=files, data=data, timeout=4)
    
    # Check if the response contains valid JSON
    if response.status_code == 200:
        try:
            print(response.json())
        except ValueError:
            print("The response did not return valid JSON:", response.text)
    else:
        print(f"Server returned status code {response.status_code} and response: {response.text}")

except requests.exceptions.Timeout:
    print("The request timed out. Please try again later.")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

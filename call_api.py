from openai import OpenAI
import base64
import os
import time

API_URL = os.environ.get("COMBATVLA_API_URL", "https://<your-server-ip>:8000/v1")
API_KEY = os.environ.get("COMBATVLA_API_KEY", "your_api_key")
MODEL_NAME = os.environ.get("COMBATVLA_MODEL", "CombatVLA")
TRUNC_TOKENS = ("<TRUNC>", "<trunc>", "<tool_call>", "</tool_call>")

def encode_image_to_base64(image_path_list):
    encoded_image_list = []
    for image_path in image_path_list:
        with open(image_path, "rb") as image_file:
            encoded_image_list.append(base64.b64encode(image_file.read()).decode('utf-8'))
    return encoded_image_list

def get_response(encoded_image_list):
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_URL,
    )

    msgs = [
                {   
                    'role': 'system', 
                    'content': 'You are a good player for the computer game: Black Myth: Wukong.'
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Assuming you are a player of the game \"Black Myth: Wukong,\" your task is to defeat the enemies in the game. Please predict the next actions based on the frame sequence. The explanation following the <TRUNC> symbol correspond one-to-one with the actions preceding it."
                        }
                    ]
                },
    ]

    for encoded_image in encoded_image_list:
        image_url = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded_image}",
            }
        }
        msgs[-1]['content'].append(image_url)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        stream=True,
        messages=msgs,
        temperature=1.0,
        max_tokens=8192,
    )


    action_json = ""
    for chunk in completion:
        if chunk.choices:
            content = chunk.choices[0].delta.dict().get("content")
            if not content:
                continue

            # The model sometimes outputs tool-call tokens as the truncation marker.
            stop_positions = [content.find(token) for token in TRUNC_TOKENS if token in content]
            if stop_positions:
                action_json += content[: min(stop_positions)]
                break

            action_json += content
    return action_json

def call_combatvla(image_path_list):
    encoded_image_list = encode_image_to_base64(image_path_list)
    action_json = get_response(encoded_image_list)

    return action_json


if __name__ == '__main__':
    image_path_1 = "./test_images/00089.png"
    image_path_2 = "./test_images/00090.png"
    image_path_3 = "./test_images/00091.png"
    image_path_4 = "./test_images/00092.png"

    image_path_list = [image_path_1, image_path_2, image_path_3]
    start_time = time.time()

    encoded_image_list = encode_image_to_base64(image_path_list)

    action_json = get_response(encoded_image_list)
    end_time = time.time()
    
    print(action_json)
    print("Inference time: " + str(end_time - start_time))

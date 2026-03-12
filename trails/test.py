from ray_cluster_access.sv_inference_api import sv_openai_completion, _convert_image_to_base64
# Works with a local file path or a URL
image_b64 = _convert_image_to_base64("https://hips.hearstapps.com/hmg-prod/images/lm25-gravity-cross-ext-8906-1-689df93f6fa32.jpg")

try:
    response = sv_openai_completion(
        model="Qwen/Qwen3.5-27B",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify the vehicle shown in the image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                ],
            }
        ],
        max_tokens=1000,
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(type(e).__name__, e)

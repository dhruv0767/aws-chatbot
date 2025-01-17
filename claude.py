import boto3
import json

prompt_data="""
You are a highly intelligent and professional chatbot designed to answer questions about Dhruv Shah, his projects, career, and expertise. 
Use the given context to deliver detailed, concise, and professional answers. Always conclude by suggesting three additional related questions 
to engage the user further. The questions should be in bulleted points. If the information is unavailable, say "I don't know" without speculating.
"""

bedrock=boto3.client(service_name="bedrock-runtime")

payload={
    "prompt":prompt_data,
    "max_tokens_to_sample":512,
    "temperature":0.8,
    "topP":0.8
}
body = json.dumps(payload)
model_id = "anthropic.claude-instant-v1"
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())
response_text = response_body.get("completions")[0].get("data").get("text")
print(response_text)
# Model overview 
DeepSeek V4-Pro
Model overview
Price
$1.74 - $0.14 - $3.48
Input - Cached - Output
Parameters
49B - 1.6T
Active - Total
Context window
1M
 
Release date
Apr 2026
 
DeepSeek V4-Pro is a 1.6T-parameter MoE model with 49B active parameters and a 1M context length. As the flagship of the DeepSeek V4 family, it features a hybrid attention architecture for efficient long-context processing and is purpose-built for advanced reasoning, complex software engineering, and long-horizon agentic tasks.
DeepSeek

# Features
License: mit
deepseek-ai/DeepSeek-V4-Pro
Supported features
Reasoning
JSON mode
Structured output
Tool calling
LoRA
Post training

# Python
import openai
import weave

# Weave autopatches OpenAI to log LLM calls to W&B
weave.init("<team>/<project>")

client = openai.OpenAI(
    # The custom base URL points to W&B Inference
    base_url='https://api.inference.wandb.ai/v1',

    # Get your API key from https://wandb.ai/authorize
    # Consider setting it in the environment as OPENAI_API_KEY instead for safety
    api_key="<your-apikey>",

    # Optional: Team and project for usage tracking
    project="<team>/<project>",
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
)

print(response.choices[0].message.content)

# CURL
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-apikey>" \
  -H "OpenAI-Project: <team>/<project>" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "Tell me a joke." }
    ]
  }'
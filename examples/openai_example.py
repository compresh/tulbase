"""
Example: Using Tulbase with OpenAI Python SDK.

1. Start Tulbase: tulbase serve
2. Run this script: python examples/openai_example.py
"""

from openai import OpenAI

# Point to Tulbase proxy instead of OpenAI directly
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-your-real-openai-key",  # Your real key — Tulbase forwards it
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a very helpful, friendly, and professional AI assistant. You should always try to be helpful and provide clear and accurate responses. Please make sure to be polite and respectful at all times."},
        {"role": "user", "content": "I would like you to please help me understand how to use Python decorators. Could you please explain them in a simple way?"},
    ],
)

print(response.choices[0].message.content)

# Check optimization stats in response headers (if using httpx directly)
# X-Tulbase-Saved-Tokens: ~45
# X-Tulbase-Saved-Percent: ~22%

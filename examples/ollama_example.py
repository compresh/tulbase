"""
Example: Using Tulbase with Ollama (local LLM).

1. Start Ollama: ollama serve
2. Set env: TULBASE_DEFAULT_BASE_URL=http://localhost:11434/v1
3. Start Tulbase: tulbase serve
4. Run: python examples/ollama_example.py

Tulbase optimizes prompts even for local models — effectively
extending your context window by 10-20%.
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="ollama",  # Ollama doesn't need a real key
)

response = client.chat.completions.create(
    model="llama3.1",
    messages=[
        {"role": "system", "content": "You are a very helpful, knowledgeable, and friendly coding assistant. You should always provide clear and detailed explanations. Please make sure your responses are accurate and well-structured."},
        {"role": "user", "content": "Could you please help me understand the difference between Python lists and tuples?"},
    ],
)

print(response.choices[0].message.content)

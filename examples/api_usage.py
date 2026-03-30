"""
Example: Using Lightbulb API with OpenAI Python SDK

This demonstrates OpenAI compatibility and Lightbulb-specific extensions.
"""

from openai import OpenAI

# Initialize client pointing to Lightbulb API
client = OpenAI(api_key="your-api-key-here", base_url="http://localhost:8080/v1")

# Example 1: Basic chat completion (OpenAI-compatible)
print("Example 1: Basic Chat Completion")
print("-" * 50)

response = client.chat.completions.create(
    model="lightbulb-7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
    ],
    temperature=0.7,
    max_tokens=200,
)

print(response.choices[0].message.content)
print()

# Example 2: Streaming chat completion
print("Example 2: Streaming Response")
print("-" * 50)

stream = client.chat.completions.create(
    model="lightbulb-7b",
    messages=[{"role": "user", "content": "Count from 1 to 5"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# Example 3: Using Lightbulb-specific extensions
print("Example 3: Lightbulb Extensions")
print("-" * 50)

response = client.chat.completions.create(
    model="lightbulb-7b",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    # Lightbulb-specific extensions
    extra_body={
        "lightbulb": {
            "reasoning_budget": {"max_chains": 5, "max_steps": 10, "max_tokens": 1000},
            "use_knowledge_base": True,
            "metadata": {"priority": "high", "tags": ["physics", "research"]},
        }
    },
)

print(response.choices[0].message.content)
print()

# Example 4: List available models
print("Example 4: List Models")
print("-" * 50)

models = client.models.list()
for model in models.data:
    print(f"- {model.id}")
print()

# Example 5: Using Lightbulb API directly with requests
print("Example 5: Direct API Access")
print("-" * 50)

import requests

# Query knowledge base
kb_response = requests.post(
    "http://localhost:8080/v1/lightbulb/knowledge/query",
    headers={
        "Authorization": "Bearer your-api-key-here",
        "Content-Type": "application/json",
    },
    json={"query": "machine learning", "max_results": 5},
)

print(f"Knowledge base query status: {kb_response.status_code}")
print(kb_response.json())
print()

# Set reasoning budget
reasoning_response = requests.post(
    "http://localhost:8080/v1/lightbulb/reasoning/budget",
    headers={
        "Authorization": "Bearer your-api-key-here",
        "Content-Type": "application/json",
    },
    json={"max_chains": 10, "max_steps": 20, "max_tokens": 2000},
)

print(f"Set reasoning budget status: {reasoning_response.status_code}")
print()

# Admin API: Get cache statistics
admin_response = requests.get(
    "http://localhost:8080/v1/lightbulb/admin/cache/stats",
    headers={"Authorization": "Bearer your-admin-key-here"},
)

print(f"Cache stats status: {admin_response.status_code}")
print(admin_response.json())

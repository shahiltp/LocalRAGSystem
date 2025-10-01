#!/usr/bin/env python3
"""
Test script for conversation history functionality
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8001"

def test_conversation_history():
    """Test the conversation history functionality"""
    
    print("🧪 Testing Conversation History Functionality")
    print("=" * 50)
    
    # Test 1: First question
    print("\n1️⃣ First Question: 'What is 1+1?'")
    
    response1 = requests.post(f"{API_BASE_URL}/v1/chat/completions", json={
        "model": "crew-ai-rag",
        "messages": [
            {"role": "user", "content": "What is 1+1?"}
        ]
    })
    
    if response1.status_code == 200:
        data1 = response1.json()
        conversation_id = data1.get("conversation_id")
        answer1 = data1["choices"][0]["message"]["content"]
        print(f"✅ Answer: {answer1}")
        print(f"📝 Conversation ID: {conversation_id}")
    else:
        print(f"❌ Error: {response1.status_code} - {response1.text}")
        return
    
    time.sleep(2)  # Wait a bit
    
    # Test 2: Follow-up question with context
    print("\n2️⃣ Follow-up Question: 'What if it is 3 and 4?'")
    
    response2 = requests.post(f"{API_BASE_URL}/v1/chat/completions", json={
        "model": "crew-ai-rag",
        "messages": [
            {"role": "user", "content": "What if it is 3 and 4?"}
        ],
        "conversation_id": conversation_id
    })
    
    if response2.status_code == 200:
        data2 = response2.json()
        answer2 = data2["choices"][0]["message"]["content"]
        print(f"✅ Answer: {answer2}")
    else:
        print(f"❌ Error: {response2.status_code} - {response2.text}")
        return
    
    time.sleep(2)  # Wait a bit
    
    # Test 3: Another follow-up
    print("\n3️⃣ Another Follow-up: 'And what about 5+6?'")
    
    response3 = requests.post(f"{API_BASE_URL}/v1/chat/completions", json={
        "model": "crew-ai-rag",
        "messages": [
            {"role": "user", "content": "And what about 5+6?"}
        ],
        "conversation_id": conversation_id
    })
    
    if response3.status_code == 200:
        data3 = response3.json()
        answer3 = data3["choices"][0]["message"]["content"]
        print(f"✅ Answer: {answer3}")
    else:
        print(f"❌ Error: {response3.status_code} - {response3.text}")
        return
    
    # Test 4: Get conversation history
    print("\n4️⃣ Getting Conversation History")
    
    history_response = requests.get(f"{API_BASE_URL}/conversations/{conversation_id}")
    
    if history_response.status_code == 200:
        history_data = history_response.json()
        print(f"✅ Conversation History Retrieved")
        print(f"📊 Message Count: {history_data['context']['message_count']}")
        print(f"📝 Recent Messages:")
        for i, msg in enumerate(history_data['history'][-3:], 1):
            print(f"   {i}. {msg['role']}: {msg['content'][:100]}...")
    else:
        print(f"❌ Error getting history: {history_response.status_code}")
    
    # Test 5: List all conversations
    print("\n5️⃣ Listing All Conversations")
    
    list_response = requests.get(f"{API_BASE_URL}/conversations")
    
    if list_response.status_code == 200:
        list_data = list_response.json()
        print(f"✅ Found {list_data['total_count']} conversations")
        for conv in list_data['conversations']:
            print(f"   - {conv['conversation_id'][:8]}... ({conv['message_count']} messages)")
    else:
        print(f"❌ Error listing conversations: {list_response.status_code}")
    
    print("\n🎉 Conversation History Test Complete!")


def test_procurement_conversation():
    """Test with procurement-related questions"""
    
    print("\n🏢 Testing Procurement Conversation")
    print("=" * 50)
    
    # Create new conversation
    conv_response = requests.post(f"{API_BASE_URL}/conversations")
    if conv_response.status_code != 200:
        print(f"❌ Error creating conversation: {conv_response.text}")
        return
    
    conversation_id = conv_response.json()["conversation_id"]
    print(f"📝 New Conversation ID: {conversation_id}")
    
    # Question 1
    print("\n1️⃣ Question: 'Describe the steps of Procurement Strategy'")
    
    response1 = requests.post(f"{API_BASE_URL}/v1/chat/completions", json={
        "model": "crew-ai-rag",
        "messages": [
            {"role": "user", "content": "Describe the steps of Procurement Strategy"}
        ],
        "conversation_id": conversation_id
    })
    
    if response1.status_code == 200:
        data1 = response1.json()
        answer1 = data1["choices"][0]["message"]["content"]
        print(f"✅ Answer: {answer1[:200]}...")
    else:
        print(f"❌ Error: {response1.status_code}")
        return
    
    time.sleep(2)
    
    # Question 2 - Follow-up
    print("\n2️⃣ Follow-up: 'Explain the 2nd step in more detail'")
    
    response2 = requests.post(f"{API_BASE_URL}/v1/chat/completions", json={
        "model": "crew-ai-rag",
        "messages": [
            {"role": "user", "content": "Explain the 2nd step in more detail"}
        ],
        "conversation_id": conversation_id
    })
    
    if response2.status_code == 200:
        data2 = response2.json()
        answer2 = data2["choices"][0]["message"]["content"]
        print(f"✅ Answer: {answer2[:200]}...")
    else:
        print(f"❌ Error: {response2.status_code}")
    
    print("\n🎉 Procurement Conversation Test Complete!")


if __name__ == "__main__":
    print("🚀 Starting Conversation History Tests")
    print("Make sure the API server is running on http://localhost:8001")
    print()
    
    try:
        # Test basic conversation history
        test_conversation_history()
        
        # Test procurement conversation
        test_procurement_conversation()
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on http://localhost:8001")
    except Exception as e:
        print(f"❌ Error: {e}")

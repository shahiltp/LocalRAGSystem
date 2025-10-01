#!/usr/bin/env python3
"""
Conversation Memory Management
Handles conversation history and context for the RAG system
"""

import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import threading
import os

class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_conversations: int = 100, max_messages_per_conversation: int = 50):
        """
        Initialize conversation memory
        
        Args:
            max_conversations: Maximum number of conversations to keep in memory
            max_messages_per_conversation: Maximum messages per conversation
        """
        self.max_conversations = max_conversations
        self.max_messages_per_conversation = max_messages_per_conversation
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Load persistent storage if available
        self.storage_file = "conversation_memory.json"
        self.load_from_storage()
    
    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID"""
        return str(uuid.uuid4())
    
    def _cleanup_old_conversations(self):
        """Remove old conversations to maintain memory limits"""
        if len(self.conversations) <= self.max_conversations:
            return
        
        # Sort by last_updated and remove oldest
        sorted_conversations = sorted(
            self.conversations.items(),
            key=lambda x: x[1].get('last_updated', datetime.min)
        )
        
        # Remove oldest conversations
        to_remove = len(self.conversations) - self.max_conversations
        for conv_id, _ in sorted_conversations[:to_remove]:
            del self.conversations[conv_id]
    
    def _cleanup_old_messages(self, conversation_id: str):
        """Remove old messages from a conversation"""
        if conversation_id not in self.conversations:
            return
        
        messages = self.conversations[conversation_id].get('messages', [])
        if len(messages) <= self.max_messages_per_conversation:
            return
        
        # Keep only the most recent messages
        self.conversations[conversation_id]['messages'] = messages[-self.max_messages_per_conversation:]
    
    def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Start a new conversation or return existing one
        
        Args:
            conversation_id: Optional existing conversation ID
            
        Returns:
            Conversation ID
        """
        with self.lock:
            if conversation_id and conversation_id in self.conversations:
                return conversation_id
            
            # Create new conversation
            conv_id = conversation_id or self._generate_conversation_id()
            self.conversations[conv_id] = {
                'id': conv_id,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'messages': [],
                'context': {},
                'metadata': {}
            }
            
            self._cleanup_old_conversations()
            self.save_to_storage()
            return conv_id
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add a message to a conversation
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if conversation_id not in self.conversations:
                return False
            
            message = {
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            self.conversations[conversation_id]['messages'].append(message)
            self.conversations[conversation_id]['last_updated'] = datetime.now().isoformat()
            
            self._cleanup_old_messages(conversation_id)
            self.save_to_storage()
            return True
    
    def get_conversation_history(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            conversation_id: Conversation ID
            limit: Optional limit on number of messages
            
        Returns:
            List of messages
        """
        with self.lock:
            if conversation_id not in self.conversations:
                return []
            
            messages = self.conversations[conversation_id].get('messages', [])
            if limit:
                messages = messages[-limit:]
            
            return messages
    
    def get_conversation_context(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation context and summary
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Context dictionary
        """
        with self.lock:
            if conversation_id not in self.conversations:
                return {}
            
            conv = self.conversations[conversation_id]
            messages = conv.get('messages', [])
            
            # Create context summary
            context = {
                'conversation_id': conversation_id,
                'message_count': len(messages),
                'last_updated': conv.get('last_updated'),
                'recent_messages': messages[-5:] if messages else [],  # Last 5 messages
                'context_summary': self._create_context_summary(messages),
                'metadata': conv.get('metadata', {})
            }
            
            return context
    
    def _create_context_summary(self, messages: List[Dict[str, Any]]) -> str:
        """
        Create a summary of conversation context
        
        Args:
            messages: List of messages
            
        Returns:
            Context summary string
        """
        if not messages:
            return "No previous conversation context."
        
        # Get last few messages for context
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        
        context_parts = []
        for msg in recent_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:200]  # Truncate long content
            context_parts.append(f"{role}: {content}")
        
        return " | ".join(context_parts)
    
    def update_conversation_metadata(self, conversation_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update conversation metadata
        
        Args:
            conversation_id: Conversation ID
            metadata: Metadata to update
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if conversation_id not in self.conversations:
                return False
            
            self.conversations[conversation_id]['metadata'].update(metadata)
            self.conversations[conversation_id]['last_updated'] = datetime.now().isoformat()
            self.save_to_storage()
            return True
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                self.save_to_storage()
                return True
            return False
    
    def save_to_storage(self):
        """Save conversations to persistent storage"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.conversations, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save conversation memory: {e}")
    
    def load_from_storage(self):
        """Load conversations from persistent storage"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    self.conversations = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load conversation memory: {e}")
            self.conversations = {}
    
    def get_all_conversations(self) -> Dict[str, Dict[str, Any]]:
        """Get all conversations (for debugging/admin purposes)"""
        with self.lock:
            return self.conversations.copy()
    
    def clear_all_conversations(self):
        """Clear all conversations"""
        with self.lock:
            self.conversations.clear()
            self.save_to_storage()


# Global conversation memory instance
conversation_memory = ConversationMemory()


def get_conversation_memory() -> ConversationMemory:
    """Get the global conversation memory instance"""
    return conversation_memory


# Utility functions for easy access
def start_conversation(conversation_id: Optional[str] = None) -> str:
    """Start a new conversation"""
    return conversation_memory.start_conversation(conversation_id)


def add_message(conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> bool:
    """Add a message to conversation"""
    return conversation_memory.add_message(conversation_id, role, content, metadata)


def get_conversation_history(conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get conversation history"""
    return conversation_memory.get_conversation_history(conversation_id, limit)


def get_conversation_context(conversation_id: str) -> Dict[str, Any]:
    """Get conversation context"""
    return conversation_memory.get_conversation_context(conversation_id)


if __name__ == "__main__":
    # Test the conversation memory
    memory = ConversationMemory()
    
    # Test conversation
    conv_id = memory.start_conversation()
    print(f"Started conversation: {conv_id}")
    
    memory.add_message(conv_id, "user", "What is 1+1?")
    memory.add_message(conv_id, "assistant", "1+1 equals 2.")
    memory.add_message(conv_id, "user", "What if it is 3 and 4?")
    
    history = memory.get_conversation_history(conv_id)
    context = memory.get_conversation_context(conv_id)
    
    print(f"History: {history}")
    print(f"Context: {context}")

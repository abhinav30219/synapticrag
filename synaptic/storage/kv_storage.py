"""JSON key-value storage implementation"""
import os
import json
from typing import Dict, List, Optional, Set, Any
from .base import BaseKVStorage, StorageConfig

class JsonKVStorage(BaseKVStorage):
    """JSON-based key-value storage"""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._file_path = os.path.join(
            config.working_dir,
            f"kv_store_{config.namespace}.json"
        )
        self._data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file"""
        if os.path.exists(self._file_path):
            with open(self._file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_data(self):
        """Save data to JSON file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        
        with open(self._file_path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
    
    async def index_done_callback(self):
        """Save data after indexing"""
        self._save_data()
    
    async def all_keys(self) -> List[str]:
        """Get all keys"""
        return list(self._data.keys())
    
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get value by key"""
        return self._data.get(id)
    
    async def get_by_ids(
        self,
        ids: List[str],
        fields: Optional[Set[str]] = None
    ) -> List[Optional[Dict[str, Any]]]:
        """Get multiple values by keys"""
        if fields is None:
            return [self._data.get(id) for id in ids]
        
        results = []
        for id in ids:
            if id in self._data:
                # Filter fields
                filtered = {
                    k: v for k, v in self._data[id].items()
                    if k in fields
                }
                results.append(filtered)
            else:
                results.append(None)
        return results
    
    async def filter_keys(self, keys: List[str]) -> Set[str]:
        """Return keys that don't exist in storage"""
        return set(key for key in keys if key not in self._data)
    
    async def upsert(
        self,
        data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Insert or update key-value pairs"""
        # Find new entries
        new_data = {
            k: v for k, v in data.items()
            if k not in self._data
        }
        
        # Update storage
        self._data.update(data)
        
        # Save changes
        self._save_data()
        
        return new_data
    
    async def drop(self):
        """Drop all data"""
        self._data.clear()
        self._save_data()
        
        # Remove file if it exists
        if os.path.exists(self._file_path):
            os.remove(self._file_path)

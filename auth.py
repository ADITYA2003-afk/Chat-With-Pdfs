import json
import hashlib
import os
from typing import Optional, Dict

class Auth:
    def __init__(self):
        self.users_file = "users.json"
        self._ensure_users_file_exists()
    
    def _ensure_users_file_exists(self):
        if not os.path.exists(self.users_file):
            with open(self.users_file, "w") as f:
                json.dump({"users": []}, f)
    
    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _load_users(self) -> Dict:
        with open(self.users_file, "r") as f:
            return json.load(f)
    
    def _save_users(self, users_data: Dict):
        with open(self.users_file, "w") as f:
            json.dump(users_data, f, indent=4)
    
    def register_user(self, username: str, email: str, password: str) -> bool:
        users_data = self._load_users()
        
        # Check if username or email already exists
        if any(user["username"] == username or user["email"] == email 
               for user in users_data["users"]):
            return False
        
        # Add new user
        new_user = {
            "username": username,
            "email": email,
            "password": self._hash_password(password)
        }
        users_data["users"].append(new_user)
        self._save_users(users_data)
        return True
    
    def login_user(self, username: str, password: str) -> Optional[Dict]:
        users_data = self._load_users()
        hashed_password = self._hash_password(password)
        
        for user in users_data["users"]:
            if user["username"] == username and user["password"] == hashed_password:
                return {"username": user["username"], "email": user["email"]}
        return None

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

class TTSService:
    """使用本地的GPT-SoVITS的文本转语音服务"""
    
    def __init__(self):
        self.api_url = os.getenv("GPT_SOVITS_API_URL", "http://localhost:9872")
    
    def text_to_speech(self, text: str, speaker_id: str = None) -> bytes:
        """调用GPT-SoVITS API进行语音合成"""
        try:
            payload = {
                "text": text,
                "speaker_id": speaker_id,
                "language": "zh",
                "text_split_method": "cut5",
                "batch_size": 1,
                "speed": 1.0
            }
            
            response = requests.post(
                f"{self.api_url}/tts",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"GPT-SoVITS API错误: {response.text}")
                return None
                
        except Exception as e:
            print(f"语音合成失败: {e}")
            return None
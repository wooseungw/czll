import json
import os

# settings.json 파일 경로
# settings_path = os.path.expanduser('~/Library/Application Support/Ultralytics/settings.json')# macOS
# settings_path = os.path.expanduser('~/.config/Ultralytics/settings.json')  # Linux
settings_path = os.path.expanduser('C:/Users/Seungwoo/AppData/Roaming/Ultralytics/settings.json')  # Windows

# 현재 작업 디렉토리
current_dir = os.getcwd()

# settings.json 수정
settings = {
    "datasets_dir": current_dir,  # 현재 디렉토리를 기본 데이터셋 경로로 설정
    "weights_dir": "weights",
    "runs_dir": "runs"
}

# 설정 저장
os.makedirs(os.path.dirname(settings_path), exist_ok=True)
with open(settings_path, 'w') as f:
    json.dump(settings, f, indent=2)

print(f"Updated settings.json with datasets_dir: {current_dir}")
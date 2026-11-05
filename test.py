import os
from pathlib import Path

folder = Path(r"c:\Users\princ\Downloads\Programs\Crop_Recommendation System")
print("Files in folder:")
for f in folder.glob("*env*"):
    print(f" -> Name: '{f.name}' | Path: '{f}'")
    try:
        with open(f, "rb") as fp:
            print("    First 30 bytes:", fp.read(30))
    except Exception as e:
        print("    Error reading:", e)
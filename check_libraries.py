# check_libraries.py
required_libs = ["transformers", "torch", "sentencepiece", "streamlit"]
import importlib

print("🔍 Checking required Python libraries...\n")
for lib in required_libs:
    try:
        importlib.import_module(lib)
        print(f"✅ {lib} is installed.")
    except ImportError:
        print(f"❌ {lib} is NOT installed. Please install it using: pip install {lib}")

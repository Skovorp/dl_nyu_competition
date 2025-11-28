from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset('tsbpp/fall2025_deeplearning', split='train')
    
    print('''
Now run smth like this


mkdir -p /workspace/images
cd /workspace/.cache/huggingface/hub/datasets--tsbpp--fall2025_deeplearning/snapshots/7b14dd4385d982457822e8e96c5081a30da146d8

for f in cc3m_96px_part*.zip; do 7z x "$f" -o/workspace/images & done; wait
''')
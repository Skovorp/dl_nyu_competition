How to run this:
- install requirements
- add .env file like this:
```
HF_TOKEN=hf_SUPERSECRET
WANDB_API_KEY=2mega29358secret
```
- apt update && apt install -y p7zip-full
- python load_dataset.py
- python train.py
you should have like 2-5-10gb of memory to load dataset and models
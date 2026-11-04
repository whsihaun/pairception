#!/usr/bin/env bash

pip install -q --no-build-isolation ./msda
hf download IPEC-COMMUNITY/AerialVG --repo-type dataset --revision 3e158594d1788279264c8126ef3b7f25259eab69 --local-dir data/AerialVG --quiet
hf download Ideallll/AerialVG aerialvg.pth --revision c36eea8f4e5dad9e4338a3c627262d23b91c8785 --local-dir checkpoints --quiet
echo "f563437fd0b6f3a2e8e1b037ce15ef6ad40c81aead385fb81dae01c7953cc812  checkpoints/aerialvg.pth" | sha256sum -c --quiet
hf download google-bert/bert-base-uncased config.json model.safetensors tokenizer.json tokenizer_config.json vocab.txt --revision 86b5e0934494bd15c9632b12f734a8a67f723594 --local-dir cache/bert-base-uncased --quiet
python -m pairception.parse

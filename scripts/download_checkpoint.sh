#!/usr/bin/env bash

curl -sSL -o checkpoints.tar.gz https://github.com/whsihaun/pairception/releases/download/v1.0/pairception-checkpoints-v1.0.tar.gz
echo "ac544a40d64d8ff3f681dc77d5a9940f9d7c96b22f7d402bd0b64aad978a05ab  checkpoints.tar.gz" | sha256sum -c --quiet
tar -xzf checkpoints.tar.gz
rm checkpoints.tar.gz

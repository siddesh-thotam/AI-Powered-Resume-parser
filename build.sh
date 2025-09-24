#!/usr/bin/env bash
# build.sh

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model (this should happen automatically via the requirements.txt URL)
python -c "import spacy; spacy.load('en_core_web_sm')"
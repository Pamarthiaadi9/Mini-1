# Improved Mental Health Forecasting by Cause Extraction of Emotions in Client-Therapist Conversations

This project proposes a new deep learning system for enhanced mental health prediction based on extracting causes of emotions and the emotions themselves from transcripts of therapy sessions. Rule-based NLP methods are applied for the emotion-cause extraction, while a classifier in the form of LSTM is utilized for predicting the status of mental health with greater explainability. It has a performance of 95.12% accuracy with considerable potential to indicate at-risk people based on language and emotional patterns.

# Overview
Conventional mental health prediction models fail to consider context underlying emotional utterances. Considering both what is expressed as emotion and why are important for psychological assessment.

The project suggests a hybrid NLP-deep learning solution that:

Identifies the cause and expressed emotion using rule-based approach via SpaCy dependency parsing.

Reduces fine-grained emotions into six broad categories: anger, disgust, fear, joy, sadness, surprise.

Calculates emotion ratios by client transcript to represent emotional states.

Utilizes an LSTM model to label the mental health status of an individual as 'Normal' or 'At Risk' based on distributions of emotions.

# Dataset
The dataset includes:

Two anonymized therapist-client transcript sets: Therapist 1 and Therapist 2.

There are several dialogues (turns) between therapists and clients in each transcript.

# Extracted Elements:

Emotion labels

Causes (text spans that are responsible for emotion)

Mapped primary emotions (6-class)

One-hot encoded emotion vectors

Emotion ratios per transcript

Emotion Extraction Pipeline
Rule-based Emotion-Cause Extraction

Utilizes predefined emotion lexicons.

Applies SpaCy's dependency parser to identify linguistic reasons for emotions.

Emotion Mapping

Translates raw emotions (e.g., "happy", "disgusted") to core emotion categories.

One-Hot Encoding + Emotion Ratios

Each transcript is one-hot encoded into a 6-element vector of emotion frequencies.

Normalized to calculate emotion distribution ratios.

Model Architecture
Input Layer

Accepts a 6-dimensional emotion ratio vector from each client.

LSTM Layer

Encodes sequential emotional dynamics across sessions (if any).

Dropout Layer

Avoids overfitting and regularizes the network.

Dense Layer + Sigmoid

# Outputs binary classification: Normal vs At Risk.

# Metric	Therapist 1	Therapist 2
Accuracy	95.12%	90.66%
Precision	95.56%	92.33%
Recall	95.10%	90.00%
F1-Score	95.32%	91.15%

# Usage-of-LLM-Nutrition-Exercise
Examples of the usage of large language models for nutrition and exercise


This project demonstrates the use of large language models (LLMs) for nutrition and exercise. It uses the Tavily API to create a dataset, fine-tunes the facebook/bart-base model with this dataset, and provides a script to interact with the fine-tuned model.

Overview:

create_dataset.py gathers data from the Tavily API to create a dataset about nutrition and exercise.
clean_dataset.py processes the dataset to remove unnecessary information.
fine_tune_model.py fine-tunes the facebook/bart-base model using the cleaned dataset.
test_model.py lets you interact with the fine-tuned model. Enter queries related to nutrition and exercise to get responses generated by the model.

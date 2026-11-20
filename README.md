🎬 Movie Review Mood Checker

A web app that uses a BERT-family model to read movie reviews and decide whether each one is Positive or Negative, with a confidence score.

Built as a mini project to apply what I learned in two NVIDIA Deep Learning Institute courses:

Building Transformer-Based Natural Language Processing Applications
Fundamentals of Accelerated Data Science

🔗 Live demo: your-app.streamlit.app (replace with your Streamlit link)

✨ Features
Tab	What it does
✍️ Single review	Type any review and get its mood, a confidence percentage, and the tokens BERT reads
📊 Many reviews	Analyse up to 2,000 real IMDB reviews, or upload your own CSV, then see accuracy, speed, charts of the top words, and download the results
🧠 How it works	A plain-English explanation of each step
🧠 How it works
Review text → Tokenizer → DistilBERT → Logits → Softmax → Positive / Negative + confidence %
Tokenizer splits the review into subword tokens (WordPiece) and turns them into ID numbers.
DistilBERT, a smaller and faster version of BERT, uses self-attention to understand each word in the context of the whole review.
A classifier head outputs two raw scores (logits): Negative and Positive.
Softmax turns those scores into percentages that add up to 100%.

Model: distilbert-base-uncased-finetuned-sst-2-english
Dataset: IMDB movie reviews (50,000 reviews labelled positive or negative)

🔗 How it connects to the NVIDIA courses
Part of the project	Course	Concept
Tokenizer, BERT, softmax	Transformer-Based NLP	Subword tokenization, attention, classification
Reading reviews in batches with FP16	Transformer-Based NLP	Batching and mixed precision (the idea behind Triton's dynamic batching)
Grouping and string analysis of results	Accelerated Data Science	DataFrame operations, groupby, string methods
cuDF vs pandas speed test	Accelerated Data Science	GPU-accelerated DataFrames

The app automatically uses cuDF (GPU) when it runs on a machine with an NVIDIA GPU and RAPIDS installed, and falls back to pandas (CPU) otherwise. The full GPU benchmark (50,000 reviews, cuDF vs pandas) is in the Colab notebook.

📈 Results (from the Colab notebook on a T4 GPU)
Metric	Result
Reviews analysed	50,000
BERT speed (FP16, batch 128)	__ reviews per second
Accuracy vs human labels	__ %
F1 score	__
cuDF speed-up over pandas (500,000 reviews)	__ x faster

(Fill these in with your own results.)

🚀 Run it yourself

1. Clone the repo

bash
git clone https://github.com/XanthusL/Movie-Review-Mood-Checker.git
cd Movie-Review-Mood-Checker

2. Install the libraries

bash
pip install -r requirements.txt

3. Start the app

bash
streamlit run app.py

It opens at http://localhost:8501. The first load takes about 30 seconds while the model downloads.

📁 Files
├── app.py              # Streamlit app
├── requirements.txt    # Python libraries
└── README.md           # This file
⚠️ Limitations
Reviews longer than 256 tokens are cut off.
The model was trained on short movie sentences, so sarcasm and mixed opinions can fool it.
The free Streamlit Cloud server uses a CPU, so large batches are slower than on a GPU.
🛠️ Built with

Python · Streamlit · Hugging Face Transformers · PyTorch · pandas · NVIDIA RAPIDS cuDF

🙏 Acknowledgements
NVIDIA Deep Learning Institute for the course content
Republic Polytechnic, C295 Special Project elective

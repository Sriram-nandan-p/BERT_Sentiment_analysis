import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Load Dataset
file_path = "Merged_Walmart_Reviews.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Display the first few rows
print(df.head())

# 2. Preprocessing
# Filter reviews for relevant keywords
keywords = ["stress", "anxiety", "sleep"]
filtered_df = df[df['Review'].str.contains('|'.join(keywords), case=False, na=False)]

# Drop missing reviews
filtered_df = filtered_df.dropna(subset=['Review'])

# Reset index for cleaner handling
filtered_df = filtered_df.reset_index(drop=True)

# 3. Tokenization
class ReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_len):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        encoding = self.tokenizer(
            review,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

# Initialize BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_len = 128

# Create Dataset
reviews = filtered_df['Review'].tolist()
dataset = ReviewDataset(reviews, tokenizer, max_len)
dataloader = DataLoader(dataset, batch_size=16)

# 4. Load Pre-trained BERT Model
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model.eval()  # Set to evaluation mode

# 5. Predict Sentiments
sentiments = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        sentiments.extend(preds.cpu().numpy())

# Map Sentiment Scores
sentiment_mapping = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive"
}

filtered_df["sentiment"] = [sentiment_mapping[s] for s in sentiments]


# 6. Visualization
sentiment_counts = filtered_df["sentiment"].value_counts()

plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind="bar", color=["red", "blue", "green"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# 7. Save Results
filtered_df.to_csv("/mnt/data/Filtered_Sentiment_Analysis.csv", index=False)
print("Analysis saved as Filtered_Sentiment_Analysis.csv")

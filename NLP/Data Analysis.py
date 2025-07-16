import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
from collections import Counter

# ------------------------part 1----------------------------
train_data = pd.read_csv(rf"C:\Users\Asus\Desktop\train_data.csv")


train_data['overall'].value_counts().sort_index().plot(kind='bar')
plt.title('Overall')
plt.xlabel('vote')
plt.ylabel('count')
plt.show()

# ------------------------part 2----------------------------

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



positive_reviews = train_data[train_data['overall'].isin([4, 5])]['reviewText'].str.cat(sep=' ')
neutral_reviews = train_data[train_data['overall'] == 3]['reviewText'].str.cat(sep=' ')
negative_reviews = train_data[train_data['overall'].isin([1, 2])]['reviewText'].str.cat(sep=' ')


def preprocess_reviewText(reviewText):
    print('-'*30 ,"preprocess_reviewText", '-'*30)

    reviewText = re.sub(r'[^\w\s]', '', reviewText)
    reviewText = re.sub(r'\d+', '', reviewText)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(reviewText.lower())
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def get_top_words(reviewText, n=10):
    print('-'*30 ,"get_top_words", '-'*30)

    tokens = preprocess_reviewText(reviewText)
    word_counts = Counter(tokens)
    return word_counts.most_common(n)


top_positive = get_top_words(positive_reviews)
top_neutral = get_top_words(neutral_reviews)
top_negative = get_top_words(negative_reviews)


def plot_bar_chart(word_counts, title):
    print('-'*30 ,"plot_bar_chart", '-'*30)

    words, counts = zip(*word_counts)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.show()


plot_bar_chart(top_positive, 'Top Words in Positive Reviews')
plot_bar_chart(top_neutral, 'Top Words in Neutral Reviews')
plot_bar_chart(top_negative, 'Top Words in Negative Reviews')



# ------------------------part 3----------------------------



group_ = train_data.groupby("reviewerName")["overall"].sum().sort_values(ascending=False)[2:12]

group_ = group_.reset_index(name='count')
print(group_)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

scatter = plt.scatter(x=group_['reviewerName'],
                      y=group_['count'],
                      s=group_['count'],
                      alpha=0.7)
plt.xlabel('Reviewer Name', fontsize=12)
plt.ylabel('Vote', fontsize=12)
plt.grid(True)

plt.bar(x=group_['reviewerName'],
        height=group_['count'],
        color='darkcyan')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()



# ------------------------part 4----------------------------



train_data = pd.read_csv(rf"C:\Users\Asus\Desktop\train_data.csv")
train_data['len_reviewText'] = train_data['reviewText'].apply(lambda x: len(x))
print(train_data)

plt.figure(figsize=(10, 5))
plt.hist(train_data["len_reviewText"], bins=500, color="skyblue", edgecolor="black")
plt.title("Histogram of Review Text Lengths (Original)")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


q1 = train_data["len_reviewText"].quantile(0.05)
q3 = train_data["len_reviewText"].quantile(0.95)


filtered_data = train_data[(train_data["len_reviewText"] >= q1) & (train_data["len_reviewText"] <= q3)]


plt.figure(figsize=(10, 5))
plt.hist(filtered_data["len_reviewText"], bins=40, color="lightgreen", edgecolor="black")
plt.title("Histogram of Review Text Lengths (Filtered)")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



# ------------------------part 5----------------------------




train_data = train_data[train_data["overall"] == 5]

top_product = train_data.groupby("asin")["overall"].sum().sort_values(ascending=False)[:10]
top_product = top_product.reset_index(name='count')
brand_title = pd.read_csv(rf"C:\Users\Asus\Desktop\title_brand.csv")
merge_df = pd.merge(top_product, brand_title, left_on='asin', right_on='asin', how='left')


merge_df['product + brand'] = merge_df["title"] + merge_df["brand"]
df = merge_df[['product + brand', 'count']]

plt.bar(x=merge_df['title'],
        height=merge_df['count'],
        color='darkcyan')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.bar(x=merge_df['brand'],
        height=merge_df['count'],
        color='darkcyan')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
from collections import Counter

# ------------------------part 1----------------------------
train_data = pd.read_csv(rf"C:\Users\Asus\Desktop\train_data.csv")


train_data['overall'].value_counts().sort_index().plot(kind='bar')
plt.title('Overall')
plt.xlabel('vote')
plt.ylabel('count')
plt.show()

# ------------------------part 2----------------------------

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



positive_reviews = train_data[train_data['overall'].isin([4, 5])]['reviewText'].str.cat(sep=' ')
neutral_reviews = train_data[train_data['overall'] == 3]['reviewText'].str.cat(sep=' ')
negative_reviews = train_data[train_data['overall'].isin([1, 2])]['reviewText'].str.cat(sep=' ')


def preprocess_reviewText(reviewText):
    print('-'*30 ,"preprocess_reviewText", '-'*30)

    reviewText = re.sub(r'[^\w\s]', '', reviewText)
    reviewText = re.sub(r'\d+', '', reviewText)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(reviewText.lower())
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def get_top_words(reviewText, n=10):
    print('-'*30 ,"get_top_words", '-'*30)

    tokens = preprocess_reviewText(reviewText)
    word_counts = Counter(tokens)
    return word_counts.most_common(n)


top_positive = get_top_words(positive_reviews)
top_neutral = get_top_words(neutral_reviews)
top_negative = get_top_words(negative_reviews)


def plot_bar_chart(word_counts, title):
    print('-'*30 ,"plot_bar_chart", '-'*30)

    words, counts = zip(*word_counts)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.show()


plot_bar_chart(top_positive, 'Top Words in Positive Reviews')
plot_bar_chart(top_neutral, 'Top Words in Neutral Reviews')
plot_bar_chart(top_negative, 'Top Words in Negative Reviews')



# ------------------------part 3----------------------------



group_ = train_data.groupby("reviewerName")["overall"].sum().sort_values(ascending=False)[2:12]

group_ = group_.reset_index(name='count')
print(group_)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

scatter = plt.scatter(x=group_['reviewerName'],
                      y=group_['count'],
                      s=group_['count'],
                      alpha=0.7)
plt.xlabel('Reviewer Name', fontsize=12)
plt.ylabel('Vote', fontsize=12)
plt.grid(True)

plt.bar(x=group_['reviewerName'],
        height=group_['count'],
        color='darkcyan')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()



# ------------------------part 4----------------------------



train_data = pd.read_csv(rf"C:\Users\Asus\Desktop\train_data.csv")
train_data['len_reviewText'] = train_data['reviewText'].apply(lambda x: len(x))
print(train_data)

plt.figure(figsize=(10, 5))
plt.hist(train_data["len_reviewText"], bins=500, color="skyblue", edgecolor="black")
plt.title("Histogram of Review Text Lengths (Original)")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


q1 = train_data["len_reviewText"].quantile(0.05)
q3 = train_data["len_reviewText"].quantile(0.95)


filtered_data = train_data[(train_data["len_reviewText"] >= q1) & (train_data["len_reviewText"] <= q3)]


plt.figure(figsize=(10, 5))
plt.hist(filtered_data["len_reviewText"], bins=40, color="lightgreen", edgecolor="black")
plt.title("Histogram of Review Text Lengths (Filtered)")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



# ------------------------part 5----------------------------




train_data = train_data[train_data["overall"] == 5]

top_product = train_data.groupby("asin")["overall"].sum().sort_values(ascending=False)[:10]
top_product = top_product.reset_index(name='count')
brand_title = pd.read_csv(rf"C:\Users\Asus\Desktop\title_brand.csv")
merge_df = pd.merge(top_product, brand_title, left_on='asin', right_on='asin', how='left')


merge_df['product + brand'] = merge_df["title"] + merge_df["brand"]
df = merge_df[['product + brand', 'count']]

plt.bar(x=merge_df['title'],
        height=merge_df['count'],
        color='darkcyan')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.bar(x=merge_df['brand'],
        height=merge_df['count'],
        color='darkcyan')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()



##
# HW1 - Sentiment Analysis
from scrapy import Selector
import requests
import nltk
from nltk import FreqDist
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt



# Scrape the website to procure data
url = "https://roboticsbiz.com/consumer-sentiment-about-ai-is-at-a-critical-tipping-point-survey/"
html = requests.get(url).content
sel = Selector(text=html)

# try to get the text from the paragraphs using css
paragraphs = sel.css('body p ::text').extract()
print(paragraphs)

# tokenize the sentences for later sentiment analysis
tokens = [nltk.sent_tokenize(sen) for sen in paragraphs]
print(tokens)

# flatten the list pulling every element from every list
# and storing it into a single list
flat_tokens = [item for elem in tokens for item in elem]
print(flat_tokens)

# tokenizer w/ regular expressions to remove punctuation
tokenizer = RegexpTokenizer(r'\w+')
t = [tokenizer.tokenize(sen) for sen in paragraphs]
print(t)
# flatten the list once more
t2 = [item for elem in t for item in elem]
print(t2)

# bag of words for the first set of sentence tokens
words = [word.lower() for word in flat_tokens]

# bag of words for tokens w/o punctuation and stopwords
stop_words = set(stopwords.words('english'))
w2 = [word.lower() for word in t2 if not word in stop_words]

# create a frequency distribution w/ first bag of words
#fdist = FreqDist(words)
#fdist.plot(10)

# freqdist w/ scrubbed bag of words
fdist2 = FreqDist(w2)

# SAVE THE FREQDIST PLOT
fig = plt.figure(figsize = (10,4))
plt.gcf().subplots_adjust(bottom=0.30) # to avoid x-ticks cut-off
fdist = FreqDist(w2)
fdist.plot(10)
plt.show()
#fig.savefig('freqDist.jpg', bbox_inches= 'tight')


# bigrams
#--------
#bigrams = nltk.bigrams(w2)
#for bigram in bigrams:
#    print(bigram)

# Analyze that sentiment
analyzer = SentimentIntensityAnalyzer()
for sentence in tokens:
    vs = analyzer.polarity_scores(sentence)
    print(sentence, "\n", vs)



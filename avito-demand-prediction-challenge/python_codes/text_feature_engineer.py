import pandas as pd
import numpy as np
import string

## NLP Libraries
# NLTK
import nltk
nltk.download([
    'stopwords',
    'punkt',
    'averaged_perceptron_tagger', 
    'averaged_perceptron_tagger_ru',
    'maxent_ne_chunker',
    'words'
])
from nltk.corpus import stopwords
stopwordSet = set(stopwords.words('russian'))
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")

# Polyglot
from polyglot.text import Text, Chunk
from polyglot.decorators import cached_property
from polyglot.downloader import downloader
#downloader.download('LANG:ru', quiet=True);

# Disable some warnings
import warnings
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)

class modified_Text(Text):
    """ 
    Modified original polyglot:
        will return empty list for self.words and self.entities
        if the input is an empty string
    """
    @property
    def words(self):
        try:
            return self.tokens
        except:
            return []

    @cached_property
    def entities(self):
        """Returns a list of entities for this blob."""
        if self.words == []:
            return []
        start = 0
        end = 0
        prev_tag = u'O'
        chunks = []
        for i, (w, tag) in enumerate(self.ne_chunker.annotate(self.words)):
            if tag != prev_tag:
                if prev_tag == u'O':
                    start = i
                else:
                    chunks.append(Chunk(self.words[start: i], start, i, tag=prev_tag, parent=self))
                prev_tag = tag
        if tag != u'O':
            chunks.append(Chunk(self.words[start: i+1], start, i+1, tag=tag, parent=self))
        return chunks

def addTextFeature(df):
    text_title = []
    text_desc  = []
    df.title = df.title.fillna('')
    df.description = df.description.fillna('')
    for _, row in df[['title', 'description']].iterrows():
        text = modified_Text(row.title)
        text.language = 'ru'
        text_title.append(text)
        text = modified_Text(row.description)
        text.language = 'ru'
        text_desc.append(text)
    df['Text_title'] = text_title
    df['Text_desc'] = text_desc
    
    return df

def cleanWordList(textObj):
    """Get a list of words without any stop words and punctuation"""
    return [wrd for wrd in textObj.words if wrd not in stopwordSet and wrd not in string.punctuation]
    
def further_features_for_text(df):
    if 'Text_title' not in df.columns and 'Text_desc' not in df.columns:
        df = addTextFeature(df)
        
    df['Words_title'] = df['Text_title'].apply(lambda x: cleanWordList(x))
    df['Words_desc'] = df['Text_desc'].apply(lambda x: cleanWordList(x))

    ## Number of stopwords: https://en.wikipedia.org/wiki/Stop_words
    df['CNT_stopwords_title'] = df['Text_title'].apply(lambda x: len([w for w in x.words if w in stopwordSet]))
    df['CNT_stopwords_desc'] = df['Text_desc'].apply(lambda x: len([w for w in x.words if w in stopwordSet]))

    ## Number of punctuation
    df['CNT_puncs_title'] = df['Text_title'].apply(lambda x: len([w for w in x.words if w in string.punctuation]))
    df['CNT_puncs_desc'] = df['Text_desc'].apply(lambda x: len([w for w in x.words if w in string.punctuation]))

    ## Number of numeric characters
    df['CNT_numerics_title'] = df['title'].apply(lambda x: len([c for c in x if c.isnumeric()]))
    df['CNT_numerics_desc'] = df['description'].apply(lambda x: len([c for c in x if c.isnumeric()]))

    ## Number of special characters
    df['CNT_exclamation_desc'] = df.Text_desc.apply(lambda x: len([w for w in x.words if w == '!']))
    df['CNT_question_desc'] = df.Text_desc.apply(lambda x: len([w for w in x.words if w == '?']))
    df['CNT_colon_desc'] = df.Text_desc.apply(lambda x: len([w for w in x.words if w == ':']))
    df['CNT_special_title'] = df.Text_title.apply(lambda x: len([w for w in x.words if w in [':', '?', '!']]))
                                                                 
    ## Number of actual words
    df['CNT_word_title'] = df.Words_title.apply(lambda x: len(x))
    df['CNT_word_desc'] = df.Words_desc.apply(lambda x: len(x))
    
    ## Number of unique words
    df['CNT_unique_word_title'] = df.Words_title.apply(lambda x: len(set(x)))
    df['CNT_unique_word_desc'] = df.Words_desc.apply(lambda x: len(set(x)))
    df['PCENT_unique_word_title'] = df['CNT_unique_word_title']/df['CNT_word_title']
    df['PCENT_unique_word_desc'] = df['CNT_unique_word_desc']/df['CNT_word_desc']
    
    ## Number of characters
    df['CNT_char_title'] = df['title'].str.len()
    df['CNT_char_desc'] = df['description'].str.len()

    ## Average Word Length
    def avg_word(wordList):
        if len(wordList) == 0:
            return 0
        return (sum(len(word) for word in wordList) / len(wordList))
    df['AVG_word_len_title'] = df['Words_title'].apply(lambda x: avg_word(x))
    df['AVG_word_len_desc'] = df['Words_desc'].apply(lambda x: avg_word(x))

    ## Count polarity (positive, negative) (http://polyglot.readthedocs.io/en/latest/Sentiment.html)
    df['CNT_Pos_Words_title'] = df['Words_title'].apply(lambda x: len([wrd for wrd in x if wrd.polarity > 0]))
    df['CNT_Pos_Words_desc'] = df['Words_desc'].apply(lambda x: len([wrd for wrd in x if wrd.polarity > 0]))
    df['CNT_Neg_Words_title'] = df['Words_title'].apply(lambda x: len([wrd for wrd in x if wrd.polarity < 0]))
    df['CNT_Neg_Words_desc'] = df['Words_desc'].apply(lambda x: len([wrd for wrd in x if wrd.polarity < 0]))
    
    # Postive word to negative word count ratio
    delta = 0.01
    df['RATIO_PosNeg_Word_title'] = (df['CNT_Pos_Words_title'] + delta) / (df['CNT_Neg_Words_title'] + delta)
    df['RATIO_PosNeg_Word_desc'] = (df['CNT_Pos_Words_desc'] + delta) / (df['CNT_Neg_Words_desc'] + delta)
    
    ## Percentage of positive and negative word count in total number of words
    df['PCENT_Pos_Word_title'] = df['CNT_Pos_Words_title'] / df['CNT_word_title']
    df['PCENT_Neg_Word_title'] = df['CNT_Neg_Words_title'] / df['CNT_word_title']
    df['PCENT_Pos_Word_desc'] = df['CNT_Pos_Words_desc'] / df['CNT_word_desc']
    df['PCENT_Neg_Word_desc'] = df['CNT_Neg_Words_desc'] / df['CNT_word_desc']

    ## Named entity
    df['CNT_Named_Entity_title'] = df['Text_title'].apply(lambda x: len([x2 for x1 in x.entities for x2 in x1]))
    df['CNT_Named_Entity_desc'] = df['Text_desc'].apply(lambda x: len([x2 for x1 in x.entities for x2 in x1]))
    
    ## Handle all columns whose values are percentage
    for col in df.columns:
        if "PCENT" in col:
            df[col] = df[col].replace(np.inf, 1).fillna(0)
            
    ## Drop text columns
    df.drop(['title', 'description', 'Text_title', 'Text_desc', 'Words_title', 'Words_desc'], axis=1, inplace=True)
    return df

def word_stem_collect(df):
    ## Get Text title/description
    if 'Text_title' not in df.columns and 'Text_desc' not in df.columns:
        df = addTextFeature(df)

    ## Get Word title/description
    df['Words_title'] = df.Text_title.apply(lambda x: cleanWordList(x))
    df['Words_desc'] = df.Text_desc.apply(lambda x: cleanWordList(x))

    ## Collect stem words from title/description
    df['Stem_title'] = df.Words_title.apply(
                                lambda x : [stemmer.stem(w) for w in x])
    df['Stem_desc'] = df.Words_desc.apply(
                                lambda x : [stemmer.stem(w) for w in x])

    ## Drop text columns
    df.drop(['title', 'description', 'Text_title', 'Text_desc', 'Words_title', 'Words_desc'], axis=1, inplace=True)
    return df

def word_ranking_scores(args):
    try:
        df, title_rank, desc_rank = args
    except:
        print(f'Need to parse title and desc rank dictionaries')
        exit()

    ## Get Text title/description
    if 'Stem_title' not in df.columns and 'Stem_desc' not in df.columns:
        df = word_stem_collect(df)

    def get_ranking_score(col, rank_dict):
        freqs = map(lambda x: rank_dict[x] if x in rank_dict.keys() else 0, col)
        return sum(set(list(freqs)))

    df['RankingScore_title'] = df.Stem_title.apply(get_ranking_score, rank_dict=title_rank)
    df['RankingScore_desc'] = df.Stem_desc.apply(get_ranking_score, rank_dict=desc_rank)
                            
    ## Drop text columns
    df.drop(['Stem_title', 'Stem_desc'], axis=1, inplace=True)
    return df


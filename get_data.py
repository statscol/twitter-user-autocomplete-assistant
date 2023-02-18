import pandas as pd
from datasets import load_dataset
import snscrape.modules.twitter as sntwitter

query = "(from:petrogustavo) until:2010-01-01 since:2006-01-01"
tweets = []
limit = 10000

if __name__=="__main__":

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date, tweet.username, tweet.content])
        
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet']).to_csv("tweets.csv",sep=";",index=False)


    data=load_dataset("csv",data_files="tweets.csv",delimiter=";",split="train") ##also could use datasets.Dataset.from_pandas(df)
    data.push_to_hub("jhonparra18/petro-tweets",token="<YOUR_HUGGINGFACE_TOKEN>")   ##if you have used huggingface-cli login before ignore the token arg  
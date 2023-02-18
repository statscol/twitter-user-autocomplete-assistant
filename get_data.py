import pandas as pd
from datasets import load_dataset
import snscrape.modules.twitter as sntwitter

twitter_handle="<YOUR_DESIRED_TWITTER_STAR>"
query = f"(from:{twitter_handle}) until:2023-01-01 since:2006-01-01"
tweets = []
limit = 60000

if __name__=="__main__":

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date, tweet.username, tweet.content])
        
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet']).to_csv("tweets.csv",sep=";",index=False)


    data=load_dataset("csv",data_files="tweets.csv",delimiter=";",split="train") ##also could use datasets.Dataset.from_pandas(df)
    data.push_to_hub(f"{<YOUR_HF_USERNAME>}/{twitter_handle.lower()}-tweets",use_auth_token="<YOUR_HUGGINGFACE_TOKEN>")   ##if you have used huggingface-cli login before ignore the token arg  
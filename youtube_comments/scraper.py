from apiclient.discovery import build
import pandas as pd
import copy as cp
# Scraper scrapes from newest comments to oldest

api_key = "" # your google youtube comment API key goes here
youtube = build('youtube', 'v3', developerKey=api_key)

def scraper(link, df=pd.DataFrame(), next_page_token=None, total=100):

    temp_total = cp.deepcopy(total)
    # ID = link[32:len(link)]
    ID = link[32:43]
    maxResults = str(temp_total)
    
    if temp_total > 100:
        maxResults = '100'

    if next_page_token == None:
        data = youtube.commentThreads().list(part='snippet', videoId=ID, maxResults=maxResults, textFormat="plainText").execute()
    else:
        data = youtube.commentThreads().list(part='snippet', videoId=ID, pageToken=next_page_token, maxResults=maxResults, textFormat="plainText").execute()
    
    if df.empty == True:
        df = pd.DataFrame(columns=['commentor', 'comment', 'time', 'likes', 'type'])
    

    for comments in data["items"]:
        commentor = comments["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
        comment = comments["snippet"]['topLevelComment']["snippet"]["textDisplay"]
        time = comments["snippet"]['topLevelComment']["snippet"]['publishedAt']
        likes = comments["snippet"]['topLevelComment']["snippet"]['likeCount']
        typ = 'comment'
        df = df.append({'commentor' : commentor,
                        'comment' : comment,
                        'time' : time,
                        'likes' : likes,
                        'type' : typ}, ignore_index=True)
            
        reply_count = comments["snippet"]['totalReplyCount']
        if df.shape[0] >= total:
                return df[0:total]
        elif reply_count > 0:
            parent_comment = comments["snippet"]['topLevelComment']["id"]
            reply_data = youtube.comments().list(part='snippet', maxResults=maxResults, parentId=parent_comment, textFormat="plainText").execute()

            for replies in reply_data["items"]:
                commentor = replies["snippet"]["authorDisplayName"]
                comment = replies["snippet"]["textDisplay"]
                time = replies["snippet"]['publishedAt']
                likes = replies["snippet"]['likeCount']
                typ = 'reply'

                df = df.append({'commentor' : commentor,
                                'comment' : comment,
                                'time' : time,
                                'likes' : likes,
                                'type' : typ}, ignore_index=True)
            if df.shape[0] >= total:
                return df[0:total]

    temp_total = total - df.shape[0]
    if "nextPageToken" in data:
        df = scraper(link, df, data["nextPageToken"], total=total)
    return df[0:total]

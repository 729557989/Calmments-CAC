from django.shortcuts import render
from youtube_comments.scraper import scraper
from AI_model.make_pred import df_prediction
import csv


# Create your views here.
def index_html(request):
    # return html home page if no input
    if request.GET.get('yt_link') == None:
        return render(request, 'index.html', context={'dataframe':None})

    yt_link = request.GET.get('yt_link')
    total = request.GET.get('total')
    pref_neu = request.GET.get('pref_neu') # if checked, returns "neutral", unchecked returns None
    pref_neg = request.GET.get('pref_neg')
    
    if pref_neu==None and pref_neg==None: # return home page if user dont want any comments at all
        return render(request, 'index.html', context={'dataframe':None})

    try:
        df = scraper(str(yt_link), total=int(total))
    except:
        return render(request, 'index.html', context={'dataframe':None})

    values_df = df_prediction(df) # values_df[0] is the predictions made, values_df[1] is the RGB colors
    df['pred'] = values_df[0]
    df['color'] = values_df[1]

    if pref_neu == 'neutral' and pref_neg == None:
        clean_df = df.loc[df['pred'] == 'neutral'] # only want neutral comments only
        context = {'dataframe':clean_df.to_dict("records")}
        return render(request, 'index.html', context=context)

    elif pref_neu == None and pref_neg == 'negative':
        clean_df = df.loc[df['pred'] != 'neutral'] # only want negative comments only
        context = {'dataframe':clean_df.to_dict("records")}
        return render(request, 'index.html', context=context)

    elif pref_neu == 'neutral' and pref_neg == 'negative':
        context = {'dataframe':df.to_dict("records")} # want all comments
        return render(request, 'index.html', context=context)
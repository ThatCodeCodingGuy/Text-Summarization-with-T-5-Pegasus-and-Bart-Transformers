# Text-Summarization-with-T-5-Pegasus-and-Bart-Transformers
This project uses T-5, Pegasus and Bart transformers with HuggingFace library for text summarization applied on a news dataset in Kaggle.

By HuggingFace library, I use "t5-base" model of T-5, "google/pegasus-xsum" model of Pegasus and "facebook/bart-large-cnn" model of Bart transformers to summarize the news
texts in the dataset. 

I use Tensorflow framework for T-5 and Pegasus and PyTorch for Bart. 

* The link to the dataset: https://www.kaggle.com/shashichander009/inshorts-news-data

Aside from the dataset, I use a single text from a news article and apply all three transformer models to it in order to see clearly what the performances of these models are compared to each other. 


* The link to the text I use independently from the dataset: https://www.euronews.com/travel/2021/12/02/france-travel-restrictions-covid-test-required-for-uk-travellers-from-saturday

# Results:

# T-5:
'only those with 'compelling reasons' allowed to travel from uk to france. flights to popular french ski destinations up by 600 per cent since rules eased. but there is still plenty of COVID travel admin to contend with.'
# Pegasus:
'Britons heading to French ski resorts have been given the all-clear to return.'
# Bart:
'Since 18 December, only those with "compelling reasons" have been allowed to travel from Britain to France. France requires all UK travellers to present a negative COVID-19 test - either antigen or PCR - taken within 24 hours before departure. Antigen tests must be certified by a laboratory and NHS lateral flow test kits are not allowed.'

# Conclusion:
From this example, Bart seems to be the best transformer for summarization. However, it needs to be said that parameters of T-5 and Pegasus can be tweaked for higher performance.
Additionally, instead of "t5-base", other more developed models (e.g. "t5-large, "t5-3b", and "t5-11b") can be used for better performance. 

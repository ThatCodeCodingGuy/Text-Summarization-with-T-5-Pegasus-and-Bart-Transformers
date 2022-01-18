#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers sentencepiece')


# In[ ]:


# T-5 Transformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
src_text=  """ France has lifted its blanket ban on UK travellers from today, 14 January, enabling vaccinated tourists to return. The country’s tourism minister announced the news on Twitter yesterday, with more official details expected to follow today. Since 18 December, only those with "compelling reasons" have been allowed to travel from Britain to France in a bid to stem the spread of the Omicron variant. The welcome news has led to a surge of holiday bookings from those eager to hit the slopes. According to easyJet, flights to popular French ski destinations are up by 600 per cent since the rules eased, with a peak in trips to Switzerland too as double-jabbed Brits can now pass through France to resorts over the border. But there is still plenty of COVID travel admin to contend with, as we explain below. France requires all UK travellers to present a negative COVID-19 test - either antigen or PCR - taken within 24 hours before departure. Antigen tests must be certified by a laboratory and NHS lateral flow test kits are not allowed. Privately booked lateral flow test kits can be used at home though, such as the Randox lateral flow test kit. After testing, users can submit their results via an app and the results will be certified by a laboratory. Users will then receive a travel certificate to present at the airport. Arrivals from the Schengen zone only need to present a negative test, taken in the 24 hours prior to departing, if they are not fully vaccinated."""
T5_PATH = 't5-base' # The name of the t-5 model (among the others)
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Whether the process is going to be on GPU or CPU
tokenizer = T5Tokenizer.from_pretrained(T5_PATH) # Create the tokenizer
model = T5ForConditionalGeneration.from_pretrained(T5_PATH).to(device) # To download and initialize the model

inputs = tokenizer.encode("summarize: " + src_text, return_tensors="pt", max_length=512, padding='max_length', truncation=True)
summary_ids = model.generate(inputs, num_beams=int(2), no_repeat_ngram_size=3, length_penalty=2., min_length=50, max_length=150, early_stopping=True)
# num_beams >> one of the decoding methods, no_repeat_ngram_size >> related with the decoding method
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output)


# In[ ]:


# Pegasus Transformer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
src_text = """ France has lifted its blanket ban on UK travellers from today, 14 January, enabling vaccinated tourists to return. The country’s tourism minister announced the news on Twitter yesterday, with more official details expected to follow today. Since 18 December, only those with "compelling reasons" have been allowed to travel from Britain to France in a bid to stem the spread of the Omicron variant. The welcome news has led to a surge of holiday bookings from those eager to hit the slopes. According to easyJet, flights to popular French ski destinations are up by 600 per cent since the rules eased, with a peak in trips to Switzerland too as double-jabbed Brits can now pass through France to resorts over the border. But there is still plenty of COVID travel admin to contend with, as we explain below. France requires all UK travellers to present a negative COVID-19 test - either antigen or PCR - taken within 24 hours before departure. Antigen tests must be certified by a laboratory and NHS lateral flow test kits are not allowed. Privately booked lateral flow test kits can be used at home though, such as the Randox lateral flow test kit. After testing, users can submit their results via an app and the results will be certified by a laboratory. Users will then receive a travel certificate to present at the airport. Arrivals from the Schengen zone only need to present a negative test, taken in the 24 hours prior to departing, if they are not fully vaccinated. """
model_name = 'google/pegasus-xsum' # The name of the pegasus model (among the others)
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Whether the process is going to be on GPU or CPU
tokenizer = PegasusTokenizer.from_pretrained(model_name) # Create the tokenizer
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device) # To download and initialize the model
batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(device) # This will tokenize and return PyTorch tensors
translated = model.generate(**batch) # With these tensors, we get the summary
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True) # Since the output is encoded as tensors, we have to decode it
print(tgt_text)


# In[ ]:


# Bart Transformer
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
src_text = """ France has lifted its blanket ban on UK travellers from today, 14 January, enabling vaccinated tourists to return. The country’s tourism minister announced the news on Twitter yesterday, with more official details expected to follow today. Since 18 December, only those with "compelling reasons" have been allowed to travel from Britain to France in a bid to stem the spread of the Omicron variant. The welcome news has led to a surge of holiday bookings from those eager to hit the slopes. According to easyJet, flights to popular French ski destinations are up by 600 per cent since the rules eased, with a peak in trips to Switzerland too as double-jabbed Brits can now pass through France to resorts over the border. But there is still plenty of COVID travel admin to contend with, as we explain below. France requires all UK travellers to present a negative COVID-19 test - either antigen or PCR - taken within 24 hours before departure. Antigen tests must be certified by a laboratory and NHS lateral flow test kits are not allowed. Privately booked lateral flow test kits can be used at home though, such as the Randox lateral flow test kit. After testing, users can submit their results via an app and the results will be certified by a laboratory. Users will then receive a travel certificate to present at the airport. Arrivals from the Schengen zone only need to present a negative test, taken in the 24 hours prior to departing, if they are not fully vaccinated. """
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(device)
translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(tgt_text)


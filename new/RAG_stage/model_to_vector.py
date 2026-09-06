'''
ทำให้ JSON 1 chunk -> vector 1 ชุด
'''
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv

load_dotenv(r'C:\Users\TUF\OneDrive\Desktop\kmitl\ISD\isd-2026-curriculum_2_cucumber\new\.env')

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class vector_model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    def word_to_vec(self , word_bath):
        # Tokenize sentences
        encoded_input = self.tokenizer(word_bath, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        print("Sentence embeddings:")
        return sentence_embeddings
    def count_token(self,word):
        tokens = self.tokenizer.tokenize(word)
        return tokens


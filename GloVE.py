import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

class GloVE:
    def __init__(self, corpus, lr=0.01, embedding_dim=50, epochs=200, x_max=100, alpha=0.75):
        self.corpus = corpus
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.x_max = x_max
        self.alpha = alpha
        
        self.vocab, self.word2idx, self.idx2word = self._build_vocab()
        self.cooc_matrix = self._build_coocurrence_matrix(self.vocab, self.word2idx)
        
        self.W = np.random.rand(len(self.vocab), embedding_dim)
        self.W_tilde = np.random.rand(len(self.vocab), embedding_dim)
        self.b = np.random.rand(len(self.vocab))
        self.b_tilde = np.random.rand(len(self.vocab))
    
    def _build_vocab(self):
        word2idx = {}
        for sentence in self.corpus:
            doc = nlp(sentence)
            for token in doc:
                if token.is_alpha:
                    word = token.text.lower()
                    if word not in word2idx:
                        word2idx[word] = len(word2idx)
        
        idx2word = {idx: word for word, idx in word2idx.items()}
        vocab = list( word2idx.keys())
                        
        return vocab, word2idx, idx2word
    
    
    def _build_coocurrence_matrix(self, vocab, word2idx, window_size=5):
        cooc_matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
        for sentence in self.corpus:
            doc = nlp(sentence)
            sentence_words = [token.text.lower() for token in doc if token.is_alpha and token.text.lower() in word2idx]
            for i, word in enumerate(sentence_words):
                word_idx = word2idx[word]
                neighbours = sentence_words[max(0, i - window_size) : min(len(sentence_words), i + window_size + 1)]
                for neighbour in neighbours:
                    if word != neighbour:
                        neighbour_idx = word2idx[neighbour]
                        cooc_matrix[word_idx][neighbour_idx] += 1.0
                        
        return cooc_matrix
                        
    def _calc_loss(self, W, W_tilde, b, b_tilde):
        loss = 0.0
        for i in range(len(self.vocab)):
            for j in range(len(self.vocab)):
                if self.cooc_matrix[i][j] > 0:
                    weight = (self.cooc_matrix[i][j] / self.x_max) ** self.alpha if self.cooc_matrix[i][j] < self.x_max else 1
                    loss += weight * (np.dot(W[i], W_tilde[j]) + b[i] + b_tilde[j] - np.log(self.cooc_matrix[i][j])) ** 2
        return loss
    
    def train(self):
        for epoch in range(self.epochs):
            for i in range(len(self.vocab)):
                for j in range(len(self.vocab)):
                    if self.cooc_matrix[i][j] > 0:
                        weight = (self.cooc_matrix[i][j] / self.x_max) ** self.alpha if self.cooc_matrix[i][j] < self.x_max else 1
                        diff = np.dot(self.W[i], self.W_tilde[j]) + self.b[i] + self.b_tilde[j] - np.log(self.cooc_matrix[i][j])
                        
                        grad_W = weight * diff * self.W_tilde[j]
                        grad_W_tilde = weight * diff * self.W[i]
                        grad_bias = weight * diff
                        grad_bias_tilde = weight * diff
                        
                        self.W[i] -= self.lr * grad_W
                        self.W_tilde[j] -= self.lr * grad_W_tilde
                        self.b[i] -= self.lr * grad_bias
                        self.b_tilde[j] -= self.lr * grad_bias_tilde
                    
            if epoch % 10 == 0:
                loss = self._calc_loss(self.W, self.W_tilde, self.b, self.b_tilde)
                print(f"epoch {epoch}, loss: {loss}")
    
    

if __name__ == "__main__":

    corpus = [
        "the king is a man who rules a kingdom",
        "the queen is a woman who rules a kingdom",
        "the man is strong and wise",
        "the woman is graceful and intelligent",
        "the king married the queen to unite their kingdoms",
        "a queen can reign in the absence of a king",
        "the man aspired to be a king one day",
        "the woman aspired to be a queen one day",
        "king and queen often host grand ceremonies",
        "the king and the queen govern the kingdom together"
    ]
    
    model = GloVE(corpus)
    model.train()
# Neural Network Language Model
(R)NNLM is typically unsupervised neural model for retriving char/word representation or generating text. You can consider from different perspectives. If you try to learn machine translation, you can understand it well first. 

## key points
+ n-gram feature
+ stateful rnn
+ perplexity (PPL) 

http://www.forkosh.com/mathtex.cgi?\Large PPL(S) = P(w_{1}w_{2}w_{3} \cdots w_{n})^{-\frac{1}{n}}  = \sqrt[n]{\frac{1}{P(w_{1}w_{2}w_{3} \cdots w_{n})}} = \sqrt[n]{\frac{1}{\prod_{i=1}^{n}P(w_{i}|w_{1}w_{2} \cdots w_{i-1})}} 

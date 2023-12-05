import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

## CLIP (Contrastive Language-Image Pre-training) 
## : 텍스트와 이미지 간의 대조 손실을 최소화하여 이미지에 대한 텍스트 설명 또는 텍스트에 대한 이미지 검색과 같은 작업에 사용

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        ## nn.Embedding을 사용하여 주어진 어휘 크기와 임베딩 차원으로 토큰을 임베딩.
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # A learnable weight matrix encodes the position information for each token
        ## nn.Parameter를 사용하여 위치 임베딩 행렬을 학습 가능한 매개변수로 정의.
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) 
        x = self.token_embedding(tokens)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding
        
        ## 주어진 토큰 시퀀스에 대해 토큰 임베딩과 위치 임베딩을 더하고 결과를 반환
        return x


## CLIPLayer
## 1. SelfAttention 및 Feedforward Layer:
## * 순방향 신경망(feedforward neural network) : 노드 간의 연결이 순환을 형성하지 "않는" 인공 신경망.
## 1-1. Self.Attention 클래스를 사용하여 Self Attention 적용.
## 1-2. 두 개의 Layernorm 층, Feedforward 층(linear_1 및 linear_2), 
##      및 QuickGELU 활성화 함수를 사용하여 Feedforward 네트워크를 정의.

## 2. Forward:
## 2-1. Self Attention 적용 후 Feedforward Layer를 거쳐서 출력 계산.
## 2-2. 출력은 이전의 residual connection과 더해져 최종 출력.

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        # Pre-attention norm
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # Self attention
        self.attention = SelfAttention(n_head, n_embd)
        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # Feedforward layer
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # (Batch_Size, Seq_Len, Dim)
        residue = x
        
        ### SELF ATTENTION ###

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        ### FEEDFORWARD LAYER ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension. 

        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear_1(x)
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x



class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        ## CLIPEmbedding을 통해 텍스트 토큰 임베딩을 생성한 뒤, CLIPLayer를 적용.
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        ## nn.LayerNorm을 통해 최종 출력을 정규화한 뒤 반환.
        self.layernorm = nn.LayerNorm(768)
    
    ## nn.ModuleList를 사용하여 CLIPLayer를 스택.
    ## 입력된 토큰에 대해 스택된 CLIPLayer들을 순차적으로 적용하여 최종 출력을 생성.
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers: 
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)
        
        return output
    
## 해당 구성은 CLIP 모델의 텍스트 부분을 정의, 이미지에 대한 처리는 별도의 이미지 모듈 필요.
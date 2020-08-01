# Movie Recommendation


## 01. Matrix Factorization

- Matrix Factorization(MF)은 추천 시스템의 협업필터링(CF, Collaborative Filtering) 기법 중 Latent Factor 모델에 해당함
- Latent Factor 모델은 아래의 그림과 같이 User와 Item을 $K$-차원을 가지는 latent vector로 표현하는 모델임
- 그 중 MF는 User와 Item을 $K$-차원을 가지는 동일한 latent space에 매핑함  

<img src="./images/mf3.png" style="zoom:48%;" />



### Singular Vector Decomposition(SVD)

- MF 모델에서 가장 일반적인 방법 중 하나는 SVD(특이값 분해)가 있음
- 하지만, SVD는 데이터의 수가 커질수록 연산 속도가 매우 느려지는 문제가 발생함
- 또한, 대부분의 Rating Matrix($\mathbf{R}$)가 매우 sparse하기 때문에 성능이 좋지 않음

<img src="./images/svd4.png" style="zoom:48%;" />



### Probabilistic Matrix Factorization(PMF)

- 

### PMF + Bias terms



## 02. Experiment 

### MovieLens-20m Dataset

### LINE Dataset



## 03. Results



## 04. Conclusion


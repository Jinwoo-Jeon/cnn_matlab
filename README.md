#Make CNN for MNIST dataset

###Jinwoo Jeon

##1. 코드 설명 및 실행
Matlab code (repelem 함수 때문에 MATLAB 2015a 이상 버전에서만 실행 가능할 듯)  
No external libraries
####RUN_script.m

- 실행용 스크립트
- preproc_data 주석을 해제하면 데이터를 새로 만들고 init 주석을 해제하면 모델을 새로 만듬
- 학습된 모델을 불러오려는 경우 init을 주석처리하고, .mat파일을 수동으로 불러온 뒤 RUN_script.m을 실행하면 됨
- 테스트만 하려는 경우 train을 주석하고 test를 주석 해제한 후 실행  

####preproc_data.m
- MNIST.mat을 로드해서 데이터를 가공하는 스크립트
- Data augmentation 및 Mean제거 (test data에서도 train mean제거)

####init.m
- 학습에 필요한 파라미터 및 모델을 정의하는 스크립트
- opt.solver struct에 학습 파라미터가 저장되고, opt.layer struct에 모델 파라미터가 정의됨

####makeModel.m
- init.m에서 정의된 모델 파라미터를 바탕으로 weight, bias 등 학습 대상 변수들을 초기화하여 model struct 및 option을 return하는 함수
- weight의 경우 Var(1,0)*sqrt(2/n)으로 초기화
- bias의 경우 0로 초기화
- PReLU의 alpha값은 해당 layer weight로 저장되고 논문에서 사용한 값인 0.25로 초기화

####train.m
- 학습 loop를 실행하는 함수
- train data를 batch size대로 나누고 랜덤한 index를 부여함
- learning rate를 iteration이 지나면서 decay 하도록 설정 (inv)  <p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616573/f3f2aee0-43b8-11e6-8a7d-3643e2e97ad6.png'/></p>  
- forward.m 함수를 이용하여 각 layer의 output을 계산
- forward 계산결과를 이용하여 Cross Entropy (Softmax output) 나 MSE (MLP output) 를 Cost function으로 계산
- error와 각 layer outut을 이용하여 backward.m 함수를 실행 (back-propagation)
- error와 test 결과를 이용하여 그래프를 plot

####test.m
- forward.m 함수를 이용하여 test data에 대한 error rate를 return 

####drawFromMat.m
- 학습된 모델을 import 하여 cost graph와 error rate graph를 그리는 스크립트

####forward.m
- CNN model에 input batch를 넣어 각 layer의 output을 return
- Convolution layer  
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616556/f389f8fa-43b8-11e6-99d3-bcf3587dacb1.png'/>
</p>
(X, Y: 4-dim matrix, batch_size * width * height * channel)  
(W: 5-dim matrix, 1 * kernel_width * kernel_height * input_channel * output_channel)  

- MAX pooling layer
- Fully-connected layer
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616556/f389f8fa-43b8-11e6-99d3-bcf3587dacb1.png'/>
</p>
(X, Y: 4-dim matrix, batch_size * width * height * channel)  
(W: 5-dim matrix, 1 * kernel_width * kernel_height * input_channel * output_channel)

- ReLU
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616557/f38bb744-43b8-11e6-832f-168beaf289e2.png'/>
</p>
- PReLU
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616558/f38da00e-43b8-11e6-9ed6-76967f7291ee.png'/>
</p>
- Softmax

Cross Entropy 를 Cost function으로 이용하기 위해 softmax를 이용하였음
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616559/f395628a-43b8-11e6-9783-d83422ea829b.png'/>
</p>

- Dropout

rand함수로 임의의 node를 0으로 비활성화시킴

####backward.m
- forward한 결과와 label과의 에러를 이용하여 back-propagation을 수행하여 weight를 update하는 함수
- Convolution layer

본 코드에서는 activation layer를 따로 설계했으므로 다음과 같이 error가 propagate됨.
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616561/f39aa9d4-43b8-11e6-9a9b-8ac720e4448e.png'/>
</p>
(∂E/∂y는 상위 layer에서 전파된 값을 사용)

또한, weight는 다음과 같이 update됨
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616560/f3995ade-43b8-11e6-9364-9778edd317bd.png'/>
</p>
Weight Decay term:
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616562/f3b0c458-43b8-11e6-882d-53962504b8a6.png'/>
</p>
Momentum term:
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616563/f3b29242-43b8-11e6-8cdd-30fbb5283c70.png'/>
</p>
- MAX pooling layer

어디에서 온 에러인지 따로 저장은 안하고 error를 그대로 이전 layer에 전달

- Fully-connected layer

Convolution layer와 같은 방식으로 error propagate, weight update

- ReLU

ReLU Layer의 input value이 음수이면 error를 죽이고 양수이면 error를 그대로 전달

- PReLU

ReLU Layer의 input value이 음수이면 alpha*error를 전달하고 양수이면 error를 그대로 전달  
alpha는 논문에 나와있는 대로 0.25로 initialize한 뒤 다음과 같이 update 하였다.
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616564/f3b4b914-43b8-11e6-9640-adddc0b5a7d7.png'/>
</p>
- Softmax

Softmax의 Cross Entropy Cost function
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616558/f38da00e-43b8-11e6-9ed6-76967f7291ee.png'/>
</p>
와 Activation function
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616566/f3c39d58-43b8-11e6-9ae6-db647396e4a1.png'/>
</p>
의 derivative를 이용하여 다음과 같이 Error propagation 식을 얻을 수 있다.
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616897/44caaa64-43bb-11e6-83b2-88cca84da623.png'/>
</p>

- Dropout

Forward path에서 임의로 고른 zero-mask를 이용하여 꺼진 node들은 backward path에서도 error propagation이 안되도록 한다.

##2. 학습 결과 및 성능
① CPCPFRF
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616567/f3c841fa-43b8-11e6-9f89-c81a7f88abd3.png'/>
</p>
 
② CPCPFRFS
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616568/f3d7424a-43b8-11e6-9eb6-2b3f40b93d03.png'/>
</p>
 
③ CRPCRPFRFS
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616569/f3d919e4-43b8-11e6-9922-12ad3a974b6c.png'/>
</p>
 
④ CPrPCPrPFPrFS
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616570/f3db2842-43b8-11e6-862d-63fd37ce2bb7.png'/>
</p>
 
⑤, ⑥ CPrPCPrPFPrDFS, CPrPCPrPFPrDFS (2)
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616571/f3e7aebe-43b8-11e6-9e32-7a55fad49ed2.png'/>
</p>
 
⑦ CPrPDCPrPDFPrDFS
<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616572/f3f1e726-43b8-11e6-930d-0d5afc47ff28.png'/>
</p>
 


<p align="center">
<img src='https://cloud.githubusercontent.com/assets/16096001/16616973/c73a0ea4-43bb-11e6-84b6-a0e77463b9b2.png'/>
</p>
 

**최종 성능: Accuracy 99.33%**

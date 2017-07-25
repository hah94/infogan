InfoGAN  
===================  

#Main Difference with original GAN  
>Use coding feature 'c' to get disentagled representations  
By doing so, we can achieve somewhat similar to supervised learning algorithms  
(If we add categorical, random variable as c1) For example, 1~10 randint (p=0.1) for cifar-10 data  
In original GAN, optimized results can just ignore informations about 'c'  
So in InfoGAN, **Mutual Information** between 'c' and generator distribution is added in objective function  

See more in [InfoGAN](https://arxiv.org/pdf/1606.03657.pdf)

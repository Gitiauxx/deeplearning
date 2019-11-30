---
title: Notes on Adversarial Learning
date: 2019-11-26
---
This post collects notes on a neural network vulnerability known as adversarial
examples. Adversarial examples are inputs designed by 
perturbing an actual input so that the neural network
outputs a wrong answer with high confidence, although the perturbed 
input is not perceptually different from the original input. 

## What are adversarial examples?
Adversarial examples are inputs to a neural network that are maliciously crafted so that
the neural network misclassifies them with high confidence. A typical example is a neural network that categorizes images b
by animal species. The left input is a a picture that is correctly classified as a panda; the right
input is the original image perturbed by the noise shown in the middle. Although there is no 
perceptual differences between the left and right images -- they look like pandas -- the neural network
is very confident that the right input corresponds to the picture of a gibbon.

![Adversarial Example](../../../assets/img/pandas.png)*Adversarial Example (source: [Goodfellow et al.](https://arxiv.org/pdf/1412.6572.pdf))*

## Why does it matter?
The lack of robustness to small perturbations is seen as a sign that 
neural networks do not learn the true data manifold. Moreover, adversarial
attacker could use that weakness to generate visually undetectable 
input perturbations that lead to a misclassified output. Such attacks 
could have dramatic consequences when applied to system with safety or security 
concerns. Think of how problematic it would be if an attacker can paint stop signs so that 
deep learning algorithms deployed in autonomous vehicles would recognize them as
a yield sign. 

## What causes this vulnerability?
Initially, non-linearities were thought to explain the lack of robustness to adversarial 
inputs. However, [Goodfellow et al.](https://arxiv.org/pdf/1412.6572.pdf) show that 
linear perturbations of non-linear models are 
sufficient to generate misclassification. Consider an image $\textbf{x}$ represented by integer
pixel value between 0 and 255. Any perturbation within 1/255 precision should not 
lead to different model output. Consider a perturbation $\eta$ within 1/255 precision and the
resuling perturbed input $\overline{x}$.
The linear combination of the perturbed image with a weight vector $\mathbf{w}$ and $\overline{x}$, 
$$w^{T}\overline{x} = w^{T}x + w^{T}\eta$$ is optimized by choosing $\eta=\epsilon sign(w)$. 
Therefore, a $\epsilon$ perturbation on each pixel of the image $\mathbf{x}$ leads to
a total perturbation of magnitude $\epsilon m n$ after the linear combination, where 
$m$ is the average value of the weight $\mathbf{w}$ across all dimensions and $n$ is the dimension
of $\mathbf{w}$. For high dimensional problems, the resulting perturbation can be large enough to
affect the final prediction of the model. 

## How to generate adversarial examples?
We have to distinguish between:
 * _White-box_ models where the architecture and parameters values are available to the 
 adversary
 * _Black-box_ models where the adversary does not access to the parameter values of the model, but can query
 the model for inputs of his choice.
 
### Fast Gradient Method
 The Fast Gradient Method (FGM) is a simple optimization procedure that has been shown to 
 have a high success rate at generating adversarial examples (see [Goodfellow et al.](https://arxiv.org/pdf/1412.6572.pdf)). 
 Consider a white-box model with known parameters $\mathbf{\theta}$ and 
 loss function $\mathcal{L}(x, y, \mathbf{\theta})$. For a given clean input $x$
 and associated label $y$, the perturbation $\eta$ that maximizes 
 $\mathcal{L}(x + \eta, y, \mathbf{\theta})$ is given by
 
 $$ \eta = \epsilon sign(\nabla \mathcal{L}_{x}(x + \eta, y, \mathbf{\theta})),$$
 
 
where the gradients can be computed efficiently by backpropagation. The method is exact
for logistic classification and has high success rate for larger-non linear models. 

### Iterative extension of Fast Gradient Method
A natural extension og FGM is to proceed iteratively and apply the fast-gradient method to adversarial
example generated at the previous iteration:

$$ \mathbf{x}_{adv, 0} = \mathbf{x}; \: \mathbf{x}_{adv, k} = Clip_{\epsilon}(\mathbf{x}_{adv, k-1} + \alpha sign(\nabla \mathcal{L}_{x}(\mathbf{x}_{adv, k-1},  y, \mathbf{\theta}))) $$

### Robustness to transformation by cameras
FGM is a simple and cheap method that has been shown to be very effective at generating adversary example.
It can be expanded to more realistic cases where generated examples will be seen by the classifier
only through a camera (that can potentially remove the perturbations introduced in the
adversarial example).  [Kurakin et al.](https://arxiv.org/pdf/1607.02533.pdf) show that neural networks are still vulnerable to perturbed inputs that
are seen by the neural network via a third party medium (e.g. camera, sensors).
They generate adversary examples and pass them to the classifier through a cell phone camera.
Therefore, adversary replacing in the real world stop signs by their perturbed counterpart could force
an autonomous vehicle to misclassify those signs as yield signs. 

### Transferability and Attack on Black-Box Models
White-box attacks are not very realistic, since vulnerabilities can be avoided by preventing
attackers to access the model's parameters and architectures. However, 
it has been shown empirically that adversarial examples generalize to models
they were not trained against. That is, examples can be learned adversarially
against a child model and then use against its parent model if it is trained with a similar
dataset as the child model. 

In practice, the attacker queries model A (parent) for n inputs $x_{i}$ and obtained model's A
predictions $y$. The attacker trains a model B (child) to predict the queried $y_{i}$ from inputs 
$x_{i}$. Since the attacker knows the structure of model B, he can launch a white-box attack on
model B and obtained adversarial inputs $x_{i, adv}$ for any $i=1, ..., n$. **Papers** show that 
these adversarial examples can be used against the original model A with high success rate.

## Bayesian Neural Network: what does it take to make robust to adversarial examples?
The literature has developed several defenses against adversarial examples, including defense distillation
and adversarial learning --generate adversarial examples during training. In **Paper**, they explore sufficient conditions 
for neural network to be robust to adversarial example. 

Consider idealized Bayesian Neural Network that are continuous models with high confidence on the training set.
It means that around each sample in the training set , there exists a small neighborhood around which 
the BNN has high confidence. Outside of those neighborhoods, if the BNN has low confidence,
then there exists no adversarial example. It seems that the key takeaway from this result is that 
robustness to adversarial examples relates to how fast uncertainty increases as a BNN sees more 
misclassified examples and how slowly output probability decreases from training samples.




 

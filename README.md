<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true},
      jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
      extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
      TeX: {
      extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
      equationNumbers: {
      autoNumber: "AMS"
      }
    }
  });
</script>

# Project 3: Gradient Boosting the Lowess Model

## Intro:
Gradient boosting is a process by which a model is can be improved. It involves running a model and calculating the diffeerence between the observed output and the predicted output for each observation. Then by adding that residual to the new regressor, the new regressor should theoretically be better than the old one. Repeating this over mulitple trials should lead to a regression that is really close to the true regressor, and give us a low mean squared error. The following will be a demonstration of this in python. 

## Python
After importing the necessary libraries, we must define ther kernels we could use for our regression: 
```Python
# Gaussian Kernel
def Gaussian(x):
  if len(x.shape)==1:
    d = np.abs(x)
  else:
    d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*d**2))

# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
  ```
Kernels are the functions that determine how points are grouped together in locally weighted regression. Think of them as fences around each observation: if a point falls within the fence, the machine assumes they are friends and the kernel will group them together for the local regression. Different kernels can lead to different locally weighted regressions.
<figure>
<center>
<img src'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1364%2F1*AjTkd9I_2aa5P4pvXIc-jg.png&f=1&nofb=1&ipt=55d8bafe006574d11f8ace0cba97ec7f756f7d03033e8b4f136be030e339d1ec&ipo=images' 
width='500px' />
<figcaption></figcaption></center>
</figure>

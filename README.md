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
<img src='https://drive.google.com/uc?export=view&id=1H8GZbgK7BXlmS3h7ZmrmujEgKUf0IChJ' 
width='500px' />
<figcaption></figcaption></center>
</figure>

Then we define a distance function to determine the distance between points for the kernel to determine if neighboring points lie in it's fence or not.
```Python
def dist(u,v):
  if len(v.shape)==1:
    v = v.reshape(1,-1)
  d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))]) #this subtracts one value from every position in the matrix 
  return d
```

# Regression Function:
The first part of our regression function is the parameters the user will need to set.
```Python
def lowess_with_xnew(x, y, xnew,f=2/3, iter=3, intercept=True, qwerty = 6, pcaComp = 2, alpha = 0.001): 
```

Then we set the number of observations, the portion of observations to be in each linear regression, and an empty array with the same length as the number of observations to later fill with y estimates
```Python
  n = len(x) #this gets the length of the x variable, the number of observations
  r = int(ceil(f * n)) # this is the portion of n that will be used in the neigborhoods
  yest = np.zeros(n) #this gives us an array with the same number as the observations filled with zeros to be changed later
```

Then we reshape the x and y matrices and add a column fot the intercept or $$/beta_0$$
```Python
  if len(y.shape)==1: # here we reshape y into a matrix if it is not already
    y = y.reshape(-1,1)

  if len(x.shape)==1: #same for x
    x = x.reshape(-1,1)
  
  if intercept:  #if intercept is true, it adds a column of ones to the matrix
    x1 = np.column_stack([np.ones((len(x),1)),x])
  else: #otherwise it does not, this term would act as the 'b' in 'y = mx +b', or the beta_0 
    x1 = x
```

Then we set up the kernel by computing the bounds for the local neighborhoods and use our distance function to remove those outside of it and use the the kernel to normalize w for each observation. 
```Python
  h = [np.sort(np.sqrt(np.sum((x-x[i])**2,axis=1)))[r] for i in range(n)]
  #we compute the max bounds for the local neighborhoods

  w = np.clip(dist(x,x) / h, 0.0, 1.0) 
  #we take the distance function of x by itself and divide it by the max bounds of the neighborhoods
  #and then remove the values that are at those max bounds or at zero
  #tricubic represents opportunity for mroe research if you want to change kernel
  #w is a symmetric matrix
  w = (1 - w ** 3) ** 3
  #this function makes values closer to one become smaller, and values closer to zero bigger, 
  #which makes sense because values closer to the observation should have more weight in the linear regression
```

Now we loop through every observation based on the iterations we set and use the kernel to remove observations, and use redige regression with our specified alpha value to create the local regressions.
```Python
  delta = np.ones(n) #square matrix filled with ones the same length as n
  for iteration in range(iter): #loops through based on how many times we specified it to cut the outliers
    for i in range(n): #loops trhough every observation and removes outliers
      W = np.diag(delta).dot(np.diag(w[i,:])) #this is the weights for removing values
      #because w is symmetric, switching the rows and columns wont matter
      #when we multiply two diag matrices we get a diag matrix
      b = np.transpose(x1).dot(W).dot(y)
      A = np.transpose(x1).dot(W).dot(x1)
      #prediction algorithms
      A = A + alpha*np.eye(x1.shape[1]) # if we want L2 regularization (ridge)
      beta = linalg.solve(A, b) #beta is the "solved" matrix for a and b between the independent and dependent variables
      yest[i] = np.dot(x1[i],beta) #set the y estimated values

    residuals = y.ravel() - yest #calculate residuals
    #you have to ravel y here because the shape of y is (330, 1) and y is (330, ), this is a subtle error that can happen within python
    #if you subtract a vector from a normal array python returns a square matrix, which is not what we want
    #print(y.shape) 
    #print(yest.shape) 
    s = np.median(np.abs(residuals)) #median of the residuals
    delta = np.clip(residuals / (qwerty * s), -1, 1) #calculate the new array with cut outliers 
    #print(delta.shape)
    delta = (1 - delta ** 3) ** 3 #assign more importance to observations that gave you less errors and vice versa
```

Finally we compare our train data with our test data using principal component analysis
```Python
  if x.shape[1]==1:
    f = interp1d(x.flatten(),yest,fill_value='extrapolate')
    output = f(xnew)
  else:
    output = np.zeros(len(xnew))
    for i in range(len(xnew)):
      ind = np.argsort(np.sqrt(np.sum((x-xnew[i])**2,axis=1)))[:r]
      #if you dont extract principle components, you would get an infinite loop
      #use delaunay triangulation 
      pca = PCA(n_components=pcaComp)
      x_pca = pca.fit_transform(x[ind])
      tri = Delaunay(x_pca,qhull_options='QJ')
      f = LinearNDInterpolator(tri,yest[ind])
      output[i] = f(pca.transform(xnew[i].reshape(1,-1))) # the output may have NaN's where the data points from xnew are outside the convex hull of X
  if sum(np.isnan(output))>0:
    g = NearestNDInterpolator(x,y.ravel()) 
    # output[np.isnan(output)] = g(X[np.isnan(output)])
    output[np.isnan(output)] = g(xnew[np.isnan(output)])
  return output
```

# Boosted Function:
After we make the function scikit leanr compliant for the locally weighted regression, we can create a function to boost it. We specify the same paramters as before and using the process I layed out in the intro, calculate the residuals and create a better model. 
```Python
def boosted_lwr(x, y, xnew, f=1/3,iter=2,intercept=True, qwerty = 6, pcaComp = 2, alpha = 0.001):
  # we need decision trees
  # for training the boosted method we use x and y
  model1 = Lowess_AG_MD(f=f,iter=iter,intercept=intercept, qwerty = qwerty, pcaComp = pcaComp, alpha = 0.001) # we need this for training the Decision Tree
  model1.fit(x,y)
  residuals1 = y - model1.predict(x)
  model2 = Lowess_AG_MD(f=f,iter=iter,intercept=intercept, qwerty = qwerty, pcaComp = pcaComp, alpha = 0.001)
  model2.fit(x,residuals1)
  output = model1.predict(xnew) + model2.predict(xnew)
  return output 
```

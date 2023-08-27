# Compressive-Sensing-on-GPU
The goal was to generate images from measurements using an algorithm known as Compressive Sensing and increase the speed by which such images are generated. Let us represent the sensing matrix using 𝐀, the original image vector using x, and the measurement vector using y.

The goal of CS is then to find x given y and 𝐀. In theory, our measurement y must equal 𝐀x, where the dimensions of 𝐀 depend on the measurement vector y. Unless we are oversampling, the dimensions of y will always be less than the dimensions of x, and hence this is an underdetermined system. This means the equation y=𝐀x is true for an infinite number of possible x. What saves us from this problem is the property that most signals are compressible in some basis or the other. The basis we choose is the Fourier basis. With this piece of information, we can aim our search for a particular x, namely one which is the most sparse. In mathematics and computer science, we quantify the sparsity using the l1 norm, which yields a scalar value that is the sum of all the elements in the vector. In conclusion, our problem reduces to finding the x satisfying the following:  

                                        Minimize | y-𝐀x|l2< e   such that    |x|l1 is smallest		(1)

Once I formulated the problem like this, I could clearly see that this can be solved using standard Machine Learning algorithms after defining a clever loss function that can be robust to noise as well. The loss function I define to solve this problem is:

                                                                      𝐉 =| y-𝐀x|l2 + lambda*|x|l1 				(2)

Using iterative methods, the value of 𝐉 is reduced until a global minimum is reached. This is achieved using a popular ML algorithm known as Gradient Decent. The code that follows was written using the Python library PyTorch.

Here is how it works: I first initialized a zero vector  x which corresponds to our original image and has dimensions equal to the number of pixels. It is the variable with respect to which the gradient will be taken. In other words, it is the only variable in Eq 2 that will be updated with each iteration. The choice to initialize x to zeros was made with the foresight that we are interested in the most sparse solution, and such a solution will have most of its values be 0.  

I then defined an optimizer, which allows me to specify the learning rate along with other parameters that help me “tune” how efficient I want the learning of  x to be in this context. I initially used SGD with a learning rate of 0.0000001.

Then comes the for loop which defines the number of iterations the algorithm will undergo in its search for the optimum x satisfying the loss function 𝐉. With each iteration, the value of x is updated in the direction of the gradient of the loss function. This update rule ensures 𝐉 becomes smaller with each iteration. There were several parameters that needed to be fine-tuned to get a good reconstruction. These include the number of iterations, lambda, and the type of optimizer to name a few. 

For the longest time, I was using the SGD optimizer, but I observed that 𝐉 was not stable as the number of iterations increased. This was due to the updates needing to become very small as x reached its optimum value. I then decided to change the optimizer Adam, which was in fact built by UofT professors. With this, I had more control over how big the learning steps were, so I could initialize the learning steps to be big initially and using the parameters known as B1 and B2, I could control how exponentially I wanted the learning rate to decrease after a specified amount of iterations.

lambda is a parameter that controls the sparsity of x, and we want the sparsest x. However, as lambda approaches infinity, our loss function would then essentially be minimizing the l1 norm of  x which would just result in the 0 vector. It is important to keep in mind that the loss function that we want to minimize is still | y-𝐀x|l2. Hence, it is a fine balance that we must aim for while choosing lambda. lambda is a hyperparameter in my algorithm and should be tuned before each new image that needs to be processed. This is one of the downsides of this approach to compressive sensing.  Once I get x  after the iterations, I can use the inverse of our specified basis to transform it back into the original image.

The old program was written using a library in Python that specializes in convex minimization. However, the drawback of this approach was the amount of computational power and time it took to minimize the cost function as the number of measurements taken increased. 

The computational efficiency in my algorithm comes from vectorizing the code and exploiting Python's efficient vector processing abilities. On top of that, I used pytorch tensors to store all tensors including vectors and matrices, and performed tensor operations which can be accelerated on Nvdia GPU significantly. 

The general rule for the time without the GPU is O(number of iterations). So if we have 1000 iterations, it will take roughly about 1 second to run. With a GPU this is faster. With the CVXPY old code, I haven't done a run time analysis but it seems to be exponential in the number of measurements we do. This is one of the reasons the new code is a significant improvement as its run time is not dependent on how many measurements are being used to reconstruct the image. Therefore, we can even perform an oversampling analysis at the same time as an undersampling analysis. 


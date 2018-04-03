SandB|ox
===
**What** is it?
---  
A software technology framework, library, platform and movement for Machine Learning.  
Fundamentally, it introduces a representation for marrying static and dynamic graph based computing with procedural and object oriented programming(yup a lil ambitions), design patterns and standards open a world of possibilities for a vibrant community.  
It borrows ideas from some of the best frameworks, libraries, and communities that exist today, while contributing novelty of it's own.  

**Why** create it?
---
1. Modular & Composable
2. Expressive & Concise
3. Open Source
     - Package management
     - Sandblox Beach
     - Reusable & Extensible
     - Flexible
4. Scalable
5. Standardization
6. Reproducible
     - Benchmarking
     - Dataset audit
7. Continuous Integration
8. Rapid Prototyping
9. Interoperable: Marries static & dynamic graph based computing
10. Object Oriented => Modular Inheritance
11. Multiple Backends
     - Tensorflow
     - PyTorch (work in progress)
12. Out-of-the-box
     - Logging
     - Saving models
     - Hierarchical graph building inferred from code
     - Tensorboard support
     - Advanced model slicing and dicing, eg: reusing initial layers of another network, transfer-learning, etc
13. More sweg...

So you can focus on the core logic, and not about the boiler-plate

**When** will it complete?
---
Well, as soon as possible, we're looking for all the help we can get. Besides, it's for the community and so it should be by the community!

**Who**'s it for?
---
In no particular order:
Community     | Justification
---|---
Research    | General Purpose, Expressive, Rapid, Reproducibility, Benchmarking, Backend Invariant
Newbies     | Abstract, Reusability, Package Management
Developers  | Package Management, Reusability, Object oriented, Open Source
Commercial  | Scalable, CI, Maintainable, Standardised, Package Management

**How** does it work?
---
It introduces the concept of a block.  
A block is a modular unit of logic that has well defined inputs and well defined outputs.  
A block undergoes a well defined life cycle:
 - Instantiation & configuration
 - Static graph building
 - Dynamic computation

Take for instance the creation of a weird block where we concatenate two arrays, add a bias and then pass it through a fully connected layer:

    @sx.tf_function
    def foo_block(x, y, bias=tf.constant(.1)):
        concat = tf.concat([x, y], axis=0)
        biased = concat + bias
        return sx.Out.logits(tf.layers.dense(biased, 2)).bias(biased)
        
By adding the `@sx.tf_function` decorator, the function is automagically turned into a block.
This means that calling this function with parameters will create a new instance of the block.

    block = foo_block(tf.Variable([[.4, .5]]), tf.placeholder(tf.float32, [None, 2]))

And this is where things get interesting.  
Sandblox infers what arguments are required to be bound before evaluation.
In this case, the second placeholder argument obviously needs to be provided a value for execution to occur.   
Providing this value is as easy as passing it as an argument when running the block as shown below:  

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(block.run([[.3, .3]])) # 2D logits array, and 2D biases array

The order of the arguments of the run function matches the order of dynamic binding arguments (in this case only 1 such argument).

Further, the block provides convenient references to it's inputs and outputs. For instance you can evaluate just the outputs like:

    print(sess.run(block.o.my_logits, feed_dict={block.i.y: [[.3, .3]]}) # 2D logits array

This means you can safely abstract away logic for creating tensors inside blocks and not loose references to them:

    def foo_block(x=tf.Variable([[.4, .5]]), input2=tf.placeholder(tf.float32, [None, 2]), bias=tf.constant(.1))
        ...

You can even create blocks in their own name spaces:

    block(tf.Variable([[.4, .5]]), ..., props=sx.Props(scope_name='block2'))

In case of tensorflow, this allows for better organization of your graph, avoiding collision of tensor names and better visualization in tensorboard.
In case of tensorflow blocks, you can also explicitly specify props for overriding the session (`session=...`) or graph (`graph=...`) in which to perform the operations, or if you want to make the name scope unique by appending an increment (`make_scope_unique=True`).

In cases where you want to separate the instantiation stage from the static graph building stage or if you want to access the props during the static evaluation, you can use class inheritance instead of the decorator:

    class FooBlock(sx.TFFunction):
	    def build(self, x, y, bias):
            concat = tf.concat([x, y], axis=0)
            biased = concat + bias
            x = biased
            for _ in range(self.props.num_layers)
                x = tf.layers.dense(x, 2)
            return sx.Out.logits(x).bias(biased)
    block = FooBlock(num_layers=4)
    ...
    block(tf.Variable([[.4, .5]]), tf.placeholder(tf.float32, [None, 2]))
    ...

This is especially useful if you want to perform some kind of custom initialization:

    class FooBlock(sx.TFFunction):
        def __init__(self, **props):
            ...
        def build(self, ...):
            ...
	    
Classes also allow you to provide custom dynamic evaluation, allowing you to utilize dynamic compute based libraries such as PyTorch:

    class FooBlock(sx.TFFunction):
        ...
        def eval(*args, **kwargs):
            result = super(FooBlock, self).eval(*args, **kwargs)
            return result * 2 if result < .5 else resultl

Thus by following this well defined design, a lot of things can be provided out-of-the-box, so that you can focus on what's inside the block instead of what's outside block.
 
Since a block has both a static and dynamic stage, it supports both static and dynamic back-ends.  
**Chaining**: The outputs of one block could be provided as inputs to another.  
**Hierarchies**: One block could be used within another block, i.e. the output block provides it inputs and uses it's outputs.  
Imagine blocks that use hierarchies and chaining together!
For example, a layer that sequentially applies blocks:

    result = MySequentialBlock(
        tf.placeholder(...),
        ConvBlock(kernel_size=3),
        DenseBlock(size=5)
    )
    result.run(...)

Imagine building your entire model in an extremely concise, modular and composable fashion:





    model = Foo(
        Octo(
            Cat(                          ^---^
                With(...),          <{||=- @ @ -=||}>
                Dragon(...),              ).-.(
                Wings(...),              '/|||\`
            ),                             '|`  
            LOL(...),
        ),
        Octo(
            Cat(
                Without(...),
                Dragon(...),
                Wings(...)
            )
        ),
        ...
    )


**Where** does it apply?
---
 - Quick and dirty prototype
 - Reusable module to share with community
 - Large models and lots of data scaling across multiple machines and GPU clusters
 - Benchmarking your code against a standard dataset
 - Streamlined project lifecycle from model prototyping and deployment
 - Opensource version-controlled projects
 - R&D and publishing reproducible results

Inspiration
---
 - React
 - Caffe Zoo
 - npm community
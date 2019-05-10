# CheckFace

Putting a face to a name

```
Winner of Facebook Hack Melbourne 2019!

```

## Sagemaker and API endpoints
For sagemaker login use [https://cdilga.signin.aws.amazon.com/console](https://cdilga.signin.aws.amazon.com/console)

Sagemaker inference endpoint documentation with a use case similar to what we have:
[https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html)

The notebook is accessible here: [https://checkfaceinstance.notebook.us-east-1.sagemaker.aws/tree/checkface](https://checkfaceinstance.notebook.us-east-1.sagemaker.aws/tree/checkface)

We will try use the most of sagemaker we can, and usuing this toolchain might be good and easy to port the existing StyleGAN Code:
https://sagemaker.readthedocs.io/en/stable/using_tf.html

Python Code for wrapping tensorflow custom code so it can be run
https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/deploying_python.rst


## Electron

Development:

```
npm install
npm start
```

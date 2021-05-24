#!/usr/bin/env python

import numpy as np
import random


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x, gradientText):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         loss and its gradients
    x -- the point (numpy array) to check the gradient at
    gradientText -- a string detailing some context about the gradient computation

    Notes:
    Note that gradient checking is a sanity test that only checks whether the
    gradient and loss values produced by your implementation are consistent with
    each other. Gradient check passing on its own doesn’t guarantee that you
    have the correct gradients. It will pass, for example, if both the loss and
    gradient values produced by your implementation are 0s (as is the case when
    you have not implemented anything). Here is a detailed explanation of what
    gradient check is doing if you would like some further clarification:
    http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/. 
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        x[ix] += h # increment by h
        random.setstate(rndstate)
        fxh, _ = f(x) # evalute f(x + h)
        x[ix] -= 2 * h # restore to previous value (very important!)
        random.setstate(rndstate)
        fxnh, _ = f(x)
        x[ix] += h
        numgrad = (fxh - fxnh) / 2 / h

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed for %s." % gradientText)
            print("First gradient error found at index %s in the vector of gradients" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print("Gradient check passed!. Read the docstring of the `gradcheck_naive`"
    " method in utils.gradcheck.py to understand what the gradient check does.")


def grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset):
    print ("======Skip-Gram with naiveSoftmaxLossAndGradient Test Cases======")

    # first test
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = \
                skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)

    assert np.allclose(output_loss, 11.16610900153398), \
           "Your loss does not match expected loss."
    expected_gradCenterVecs = [[ 0.,          0.,          0.        ],
                               [ 0.,          0.,          0.        ],
                               [-1.26947339, -1.36873189,  2.45158957],
                               [ 0.,          0.,          0.        ],
                               [ 0.,          0.,          0.        ]]
    expected_gradOutsideVectors = [[-0.41045956,  0.18834851,  1.43272264],
                                   [ 0.38202831, -0.17530219, -1.33348241],
                                   [ 0.07009355, -0.03216399, -0.24466386],
                                   [ 0.09472154, -0.04346509, -0.33062865],
                                   [-0.13638384,  0.06258276,  0.47605228]]
                     
    assert np.allclose(output_gradCenterVecs, expected_gradCenterVecs), \
           "Your gradCenterVecs do not match expected gradCenterVecs."
    assert np.allclose(output_gradOutsideVectors, expected_gradOutsideVectors), \
           "Your gradOutsideVectors do not match expected gradOutsideVectors."
    print("The first test passed!")

    # second test
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = \
                skipgram("b", 3, ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    assert np.allclose(output_loss, 9.87714910003414), \
           "Your loss does not match expected loss."
    expected_gradCenterVecs = [[ 0.,          0.,          0.        ],
                               [-0.14586705, -1.34158321, -0.29291951],
                               [ 0.,          0.,          0.        ],
                               [ 0.,          0.,          0.        ],
                               [ 0.,          0.,          0.        ]]
    expected_gradOutsideVectors = [[-0.30342672,  0.19808298,  0.19587419],
                                   [-0.41359958,  0.27000601,  0.26699522],
                                   [-0.08192272,  0.05348078,  0.05288442],
                                   [ 0.6981188,  -0.4557458,  -0.45066387],
                                   [ 0.10083022, -0.06582396, -0.06508997]]
                     
    assert np.allclose(output_gradCenterVecs, expected_gradCenterVecs), \
           "Your gradCenterVecs do not match expected gradCenterVecs."
    assert np.allclose(output_gradOutsideVectors, expected_gradOutsideVectors), \
           "Your gradOutsideVectors do not match expected gradOutsideVectors."
    print("The second test passed!")

    # third test
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = \
                skipgram("a", 3, ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)

    assert np.allclose(output_loss, 10.810758628593335), \
           "Your loss does not match expected loss."
    expected_gradCenterVecs = [[-1.1790274,  -1.35861865,  1.53590492],
                               [ 0.,          0.,          0.        ],
                               [ 0.,          0.,          0.        ],
                               [ 0.,          0.,          0.        ],
                               [ 0.,          0.,          0.        ]]
    expected_gradOutsideVectors = [[-7.96035953e-01, -1.79609012e-02,  2.07761330e-01],
                                   [ 1.40175316e+00,  3.16276545e-02, -3.65850437e-01],
                                   [-1.99691259e-01, -4.50561933e-03,  5.21184016e-02],
                                   [ 2.02560028e-02,  4.57034715e-04, -5.28671357e-03],
                                   [-4.26281954e-01, -9.61816867e-03,  1.11257419e-01]]
                                                     
    assert np.allclose(output_gradCenterVecs, expected_gradCenterVecs), \
           "Your gradCenterVecs do not match expected gradCenterVecs."
    assert np.allclose(output_gradOutsideVectors, expected_gradOutsideVectors), \
           "Your gradOutsideVectors do not match expected gradOutsideVectors."
    print("The third test passed!")

    print("All 3 tests passed!")


def grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient):
    print ("======Skip-Gram with negSamplingLossAndGradient======")  

    # first test
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = \
                skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:],
                dummy_vectors[5:,:], dataset, negSamplingLossAndGradient)

    assert np.allclose(output_loss, 16.15119285363322), \
           "Your loss does not match expected loss."
    expected_gradCenterVecs = [[ 0.,          0.,          0.        ],
                               [ 0.,          0.,          0.        ],
                               [-4.54650789, -1.85942252,  0.76397441],
                               [ 0.,          0.,          0.        ],
                               [ 0.,          0.,          0.        ]]
    expected_gradOutsideVectors = [[-0.69148188,  0.31730185,  2.41364029],
                                   [-0.22716495,  0.10423969,  0.79292674],
                                   [-0.45528438,  0.20891737,  1.58918512],
                                   [-0.31602611,  0.14501561,  1.10309954],
                                   [-0.80620296,  0.36994417,  2.81407799]]
                     
    assert np.allclose(output_gradCenterVecs, expected_gradCenterVecs), \
           "Your gradCenterVecs do not match expected gradCenterVecs."
    assert np.allclose(output_gradOutsideVectors, expected_gradOutsideVectors), \
           "Your gradOutsideVectors do not match expected gradOutsideVectors."
    print("The first test passed!")

    # second test
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = \
                skipgram("c", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:],
                dummy_vectors[5:,:], dataset, negSamplingLossAndGradient)
    assert np.allclose(output_loss, 28.653567707668795), \
           "Your loss does not match expected loss."
    expected_gradCenterVecs = [  [ 0.,          0.,          0.        ],
                                 [ 0.,          0.,          0.        ],
                                 [-6.42994865, -2.16396482, -1.89240934],
                                 [ 0.,          0.,          0.        ],
                                 [ 0.,          0.,          0.        ]]
    expected_gradOutsideVectors = [  [-0.80413277,  0.36899421,  2.80685192],
                                     [-0.9277269,   0.42570813,  3.23826131],
                                     [-0.7511534,   0.34468345,  2.62192569],
                                     [-0.94807832,  0.43504684,  3.30929863],
                                     [-1.12868414,  0.51792184,  3.93970919]]
                     
    assert np.allclose(output_gradCenterVecs, expected_gradCenterVecs), \
           "Your gradCenterVecs do not match expected gradCenterVecs."
    assert np.allclose(output_gradOutsideVectors, expected_gradOutsideVectors), \
           "Your gradOutsideVectors do not match expected gradOutsideVectors."
    print("The second test passed!")

    # third test
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = \
                skipgram("a", 3, ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], 
                dummy_vectors[5:,:], dataset, negSamplingLossAndGradient)
    assert np.allclose(output_loss, 60.648705494891914), \
           "Your loss does not match expected loss."
    expected_gradCenterVecs = [  [-17.89425315,  -7.36940626,  -1.23364121],
                                 [  0.,           0.,           0.        ],
                                 [  0.,           0.,           0.        ],
                                 [  0.,           0.,           0.        ],
                                 [  0.,           0.,           0.        ]]
    expected_gradOutsideVectors = [[-6.4780819,  -0.14616449,  1.69074639],
                                   [-0.86337952, -0.01948037,  0.22533766],
                                   [-9.59525734, -0.21649709,  2.5043133 ],
                                   [-6.02261515, -0.13588783,  1.57187189],
                                   [-9.69010072, -0.21863704,  2.52906694]]
                                                     
    assert np.allclose(output_gradCenterVecs, expected_gradCenterVecs), \
           "Your gradCenterVecs do not match expected gradCenterVecs."
    assert np.allclose(output_gradOutsideVectors, expected_gradOutsideVectors), \
           "Your gradOutsideVectors do not match expected gradOutsideVectors."
    print("The third test passed!")

    print("All 3 tests passed!")

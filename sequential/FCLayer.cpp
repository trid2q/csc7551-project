#include "FCLayer.h"
#include <cmath>

#include "Cnn.h"
#include "Gradient.h"

namespace edu {
namespace tri {
namespace cnn {

float activationDerivative(float x) {
    float sig = 1.0f / (1.0f + exp(-x));
    return sig * (1 - sig);
}

float activationFunction(float x) {
    float sig = 1.0f / (1.0f + exp(-x));
    return sig;
}
    
void FCLayer::adjustWeights() {
    for (int m = 0; m < nNodes_; ++m) { 
        Gradient& gradient = gradients_[m];
        for (int d = 0; d < in_->depth(); ++d) {
	    Matrix<float>* mIn = in_->get(d);	
	    for (int i = 0; i < in_->height(); ++i) {
		for (int j = 0; j < in_->width(); ++j) {
                    int n = d*mIn->size() + i*mIn->width() + j;
		    float w = weights_->value(m,n,0);
		    float newWeight = gradient.adjustWeight(w, in_->value(i,j,d));
		    weights_->setValue(m,n,0, newWeight);
                }
            }
        }
	gradient.update();
    }
}
Volume<float>* FCLayer::computeError(Volume<float>* nextLayerWeights, Volume<float>* nextLayerDeltas) {
    assert(nextLayerWeights->depth() == nextLayerDeltas->height());
    assert(nextLayerDeltas->width() == nNodes_);
    for (int m = 0; m < nNodes_; ++m) { 
        Gradient& gradient = gradients_[m];
        gradient.value(nextLayerDeltas->value(0, m, 0)*activationDerivative(nodeSumIn_.value(0,m,0)));
        for (int d = 0; d < in_->depth(); ++d) {
	    Matrix<float>* mIn = in_->get(d);	
	    for (int i = 0; i < in_->height(); ++i) {
		for (int j = 0; j < in_->width(); ++j) {
                    int n = d*mIn->size() + i*mIn->width() + j;
	            (*deltas_)(i, j, d) = (*deltas_)(i, j, d) + (*weights_)(m, n, 0)*gradient.value();
                }
            }
        }
    }
    return deltas_;
}


Volume<float>* FCLayer::forward(Volume<float>* in) {
    in_ = in;
    for (int m = 0; m < nNodes_; ++m) { 
        float sum = 0.0f;
        for (int d = 0; d < in_->depth(); ++d) {
	    Matrix<float>* mIn = in_->get(d);	
	    for (int i = 0; i < in_->height(); ++i) {
		for (int j = 0; j < in_->width(); ++j) {
                    int n = d*mIn->size() + i*mIn->width() + j;
	            sum += in_->value(i, j, d) * weights_->value(m, n, 0);
                }
            }
        }
	nodeSumIn_(0, m, 0) = sum;
        (*out_)(0, m, 0) = activationFunction(sum);
    }
    return out_;
}

}
}
}

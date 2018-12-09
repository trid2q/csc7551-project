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
    int nNumInputs = in_->size();
    int nMatrixSize = in_->width()*in_->height();
    for (int m = 0; m < nNodes_; ++m) { 
        Gradient& gradient = gradients_[m];
	int i;
	#pragma omp paralell for private(i) schedule(dynamic)
        for (i = 0; i < nNumInputs; ++i) {
	    int mD = i/nMatrixSize;
            int mI = (i%nMatrixSize)/in_->width();
            int mJ = (i%nMatrixSize) % in_->width();
	    float w = weights_->value(m, i, 0);
	    float newWeight = gradient.adjustWeight(w, in_->value(mI, mJ, mD));
	    weights_->setValue(m, i, 0, newWeight);
	}
        gradient.update();
    }
}
Volume<float>* FCLayer::computeError(Volume<float>* nextLayerWeights, Volume<float>* nextLayerDeltas) {
    assert(nextLayerWeights->depth() == nextLayerDeltas->height());
    assert(nextLayerDeltas->width() == nNodes_);
    int nNumInputs = in_->size();
    int nMatrixSize = in_->width()*in_->height();
    for (int m = 0; m < nNodes_; ++m) { 
        Gradient& gradient = gradients_[m];
        gradient.value(nextLayerDeltas->value(0, m, 0)*activationDerivative(nodeSumIn_.value(0,m,0)));
	int i;
	#pragma omp paralell for private(i) schedule(dynamic)
        for (i = 0; i < nNumInputs; ++i) {
	    int mD = i/nMatrixSize;
            int mI = (i%nMatrixSize)/in_->width();
            int mJ = (i%nMatrixSize) % in_->width();
            (*deltas_)(mI, mJ, mD) = (*deltas_)(mI, mJ, mD) + (*weights_)(m, i, 0)*gradient.value();
	}
    }
    return deltas_;
}


Volume<float>* FCLayer::forward(Volume<float>* in) {
    in_ = in;
    int nNumInputs = in_->size();
    int nMatrixSize = in_->width()*in_->height();
    for (int m = 0; m < nNodes_; ++m) { 
        float sum = 0.0f;
	int i;
	#pragma omp paralell for private(i) schedule(dynamic)
        for (i = 0; i < nNumInputs; ++i) {
	    int mD = i/nMatrixSize;
            int mI = (i%nMatrixSize)/in_->width();
            int mJ = (i%nMatrixSize) % in_->width();
	    #pragma omp critical
	    sum += in_->value(mI, mJ, mD) * weights_->value(m, i, 0);
            #pragma omp end critical
	}
	nodeSumIn_(0, m, 0) = sum;
        (*out_)(0, m, 0) = activationFunction(sum);
    }
    return out_;
}

}
}
}

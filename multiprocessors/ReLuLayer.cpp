#include "ReLuLayer.h"

namespace edu {
namespace tri {
namespace cnn {

Volume<float>* ReLuLayer::forward(Volume<float>* in) {
    assert(in);
    in_ = in;

    int d, i;
    Matrix<float>* mIn = NULL;
    Matrix<float>* mOut = NULL;

    #pragma omp parallel private(d, i)
    for (d = 0; d < in_->depth(); ++d) {
        mIn = in_->get(d);
        mOut = out_->get(d);

        #pragma omp for schedule(dynamic)
	for (i = 0; i < in_->height()*in_->width(); ++i) {
	    int mI = i / in_->width();
	    int mJ = i % in_->width();
	    if ((*mIn)(mI, mJ) < 0) {
		(*mOut)(mI, mJ) = 0;
	    } else {
		(*mOut)(mI, mJ) = (*mIn)(mI, mJ);
	    }
	}
    }
    return out_;
}

Volume<float>* ReLuLayer::computeError(Volume<float>* nextLayerWeights, Volume<float>* nextLayerDeltas) {
    assert(nextLayerDeltas);
    assert(in_);

    int i, j, d;
    #pragma omp parallel private(d, i, j)
    for (d = 0; d < in_->depth(); ++d) {
         Matrix<float>* mIn = in_->get(d);
         #pragma omp for schedule(dynamic)
         for (i = 0; i < mIn->height(); ++i) {
            for (j = 0; j < mIn->width(); ++j) {
                if (mIn->value(i,j) < 0) {
                    (*deltas_)(i, j, d) = 0;
                } else {
		    (*deltas_)(i, j, d) = (*nextLayerDeltas)(i, j, d);
		}
            }
         }
    }
    return deltas_;
}

}
}
}


#include "ReLuLayer.h"

namespace edu {
namespace tri {
namespace cnn {

Volume<float>* ReLuLayer::forward(Volume<float>* in) {
    assert(in);
    in_ = in;
    for (int d = 0; d < in_->depth(); ++d) {
         Matrix<float>* mIn = in_->get(d);
         for (int i = 0; i < mIn->height(); ++i) {
            for (int j = 0; j < mIn->width(); ++j) {
                if (mIn->value(i,j) < 0) {
                    (*out_)(i, j, d) = 0;
                } else {
	            (*out_)(i, j, d) = (*mIn)(i,j);
                }
            }
         }
    }
    return out_;
}

Volume<float>* ReLuLayer::computeError(Volume<float>* nextLayerWeights, Volume<float>* nextLayerDeltas) {
    assert(nextLayerDeltas);
    assert(in_);
    for (int d = 0; d < in_->depth(); ++d) {
         Matrix<float>* mIn = in_->get(d);
         for (int i = 0; i < mIn->height(); ++i) {
            for (int j = 0; j < mIn->width(); ++j) {
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


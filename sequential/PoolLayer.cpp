#include "PoolLayer.h"

#include "Volume.h"

namespace edu {
namespace tri {
namespace cnn {

Volume<float>* PoolLayer::computeError(Volume<float>* nextLayerWeights, Volume<float>* nextLayerDeltas) {
    for (int d = 0; d < in_->depth(); ++d) {
        Matrix<float>* mDelta = deltas_->get(d);
	Matrix<float> mForwardDelta = *(nextLayerDeltas->get(d));
	Matrix<float>& mIn = *(in_->get(d));
        Matrix<float>& mOut = *(out_->get(d));

	int i, j;
	for (i = 0; i < mIn.height(); ++i) {
           int outI = i/nWinSize_;
	   for (int j = 0; j < mIn.width(); ++j) {
               int outJ = j/nWinSize_;
               if (mIn(i,j) == mOut(outI, outJ)) {
                   (*mDelta)(i, j) = mForwardDelta(outI, outJ);
	       } else {
		   (*mDelta)(i, j) = 0;
	       }
	   }
	}
    }
    return deltas_;
}

Volume<float>* PoolLayer::forward(Volume<float>* in) {
    in_ = in;
    assert(in_->width() % nWinSize_ == 0);
    assert(in_->height() % nWinSize_ == 0);
    for (int d = 0; d < in_->depth(); ++d) {
        int outI = -1;
        int outJ = -1;
	Matrix<float>* mIn = in_->get(d);
	for (int i = 0; i < mIn->height(); i += nWinSize_) {
	   ++outI;
           outJ = -1;
	   for (int j = 0; j < mIn->width(); j += nWinSize_) {
               ++outJ;
	       float fMax = numeric_limits<float>::min();
               for (int m = 0; m < nWinSize_; ++m) {
		  for (int n = 0; n < nWinSize_; ++n) {
		     fMax = ::max(fMax, mIn->value(i + m, j + n));
		  }
	       }
               (*out_)(outI, outJ, d) = fMax;
	   }
	}
    }
    return out_;
}

}
}
}


